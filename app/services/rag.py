from lib2to3.fixes.fix_input import context
from typing import Tuple, List, Optional, Any, Dict, Union
import asyncio
import logging
from pathlib import Path
from app.services.embeddings import HFEmbeddingProvider
from app.services.vector_store import FaissVectorStore
from app.services.llm import default_llm
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, 
                 emb_provider: Optional[HFEmbeddingProvider] = None, 
                 vector_store: Optional[FaissVectorStore] = None, 
                 llm_client: Any = default_llm,
                 vector_store_path: Optional[Path] = None):
        """Initialize the RAG service.
        
        Args:
            emb_provider: Provider for generating embeddings (defaults to HFEmbeddingProvider)
            vector_store: Optional pre-initialized vector store
            llm_client: The LLM client to use for generating answers
            vector_store_path: Path to the vector store directory
        """
        from app.services.embeddings import HFEmbeddingProvider
        
        # Initialize embedding provider
        self.embedding_provider = emb_provider or HFEmbeddingProvider()
        
        # Initialize vector store
        if vector_store is not None:
            self.vs = vector_store
        else:
            # Use provided path or default to data/vector_store
            vs_path = Path(vector_store_path) if vector_store_path else Path("data/vector_store")
            vs_path.mkdir(parents=True, exist_ok=True)
            self.vs = FaissVectorStore(dim=384, persist_path=str(vs_path))  # all-MiniLM-L6-v2 uses 384 dim
            
        self.llm = llm_client
        logger.info("RAG service initialized with HuggingFace LLM and embedding provider")

    async def answer_query(self, query: str, top_k: int = 5, min_score: float = 0.3) -> Tuple[str, List[Dict[str, Union[str, int, float]]]]:
        """Answer a query using the RAG pipeline.
        
        Args:
{{ ... }}
            top_k: Number of relevant contexts to retrieve
            min_score: Minimum similarity score for a context to be considered relevant
            
        Returns:
            A tuple of (answer, list of source documents)
        """
        MAX_CONTEXT_CHARS = 4000  # Reduced from 8000 to be safer
        try:
            # Get the query embedding (using embed_texts with a single query)
            query_embedding = (await self.embedding_provider.embed_texts([query]))[0]
            
            # Retrieve relevant chunks (get more than needed to have options)
            relevant_hits = self.vs.query(query_embedding, top_k=top_k * 2)
            
            # Filter by minimum score and sort by score (highest first)
            relevant_hits = sorted(
                [h for h in relevant_hits if h[1] >= min_score],
                key=lambda x: x[1], 
                reverse=True
            )
            
            if not relevant_hits:
                return "I couldn't find any relevant information to answer your question.", []
                
            # Build context within character limit
            contexts = []
            total_chars = 0
            
            for hit in relevant_hits:
                chunk_text = self._format_hit(hit)
                if not chunk_text:
                    continue
                    
                chunk_size = len(chunk_text)
                
                # Skip if adding this chunk would exceed the limit (leave room for the prompt)
                if total_chars + chunk_size > MAX_CONTEXT_CHARS and contexts:
                    break
                    
                contexts.append(chunk_text)
                total_chars += chunk_size
                
                # Stop if we've reached the maximum number of chunks
                if len(contexts) >= top_k:
                    break
            
            if not contexts:
                return "I found some information, but couldn't process it properly. Please try a different question.", []
                
            logger.info(f"Using {len(contexts)} context chunks with {total_chars} characters")
            
            # Build the prompt with the selected contexts
            prompt = self._build_prompt(query, contexts)
            
            # Generate the answer using the LLM with instructions to only use the context
            answer = await self.llm.generate(prompt)
            print("answer", answer)
            
            # Verify the answer is grounded in the context
            if not self._is_answer_grounded(answer, contexts):
                return "I'm not confident in my answer based on the available information. Could you try rephrasing your question or providing more context?", []
            
            # Prepare sources from the used contexts
            used_indices = [i for i, _ in enumerate(contexts)]
            sources = [
                {
                    'source': relevant_hits[i][2].get('source', 'Unknown source'),
                    'score': float(relevant_hits[i][1]),
                    'chunk_index': relevant_hits[i][2].get('chunk_index', -1)
                }
                for i in used_indices if i < len(relevant_hits)
            ]
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error in answer_query: {str(e)}", exc_info=True)
            return f"Error processing your question: {str(e)}", []

    def _format_hit(self, hit):
        """Extract text from a hit tuple.
        
        Args:
            hit: A tuple of (id, score, metadata)
            
        Returns:
            The text content from the hit's metadata or id_map
        """
        if not hit or len(hit) < 2:
            logger.warning(f"Invalid hit format: {hit}")
            return ""
            
        # Get the document ID from the hit
        doc_id = hit[0]
        logger.debug(f"Processing hit with ID: {doc_id}")
        
        # First try to get text from the vector store's id_map
        if hasattr(self.vs, 'id_map') and self.vs.id_map:
            logger.debug(f"id_map type: {type(self.vs.id_map)}")
            logger.debug(f"id_map first few items: {list(self.vs.id_map.items())[:2] if hasattr(self.vs.id_map, 'items') else self.vs.id_map[:2]}")
            
            # Handle both dict and list formats for id_map
            if hasattr(self.vs.id_map, 'items'):  # If it's a dict
                if doc_id in self.vs.id_map:
                    return str(self.vs.id_map[doc_id])
            else:  # If it's a list
                for item in self.vs.id_map:
                    if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] == doc_id:
                        return str(item[1])
            
        # If not found in id_map, try to get from metadata
        if len(hit) >= 3:
            metadata = hit[2]
            logger.debug(f"Hit metadata: {metadata}")
            if isinstance(metadata, dict):
                print("metadata", metadata)
                text = metadata.get('text', '')
                if text:
                    return str(text)
        
        logger.warning(f"Could not find text content for hit: {hit}")
        return ""

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        """Build a prompt that instructs the model to only use the provided context."""
        # Filter out empty contexts
        valid_contexts = [ctx for ctx in contexts if ctx.strip()]
        
        if not valid_contexts:
            logger.warning("No valid contexts provided for prompt building")
            return f"""I don't have enough information to answer this question. Please provide more context or rephrase your question.
            
            Question: {query}
            
            Answer: I don't know."""
            
        context_block = "\n\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(valid_contexts))
        
        return f"""Answer the question using ONLY the information from the provided context. 
        If the answer cannot be found in the context, respond with "I don't know".
        
        {context_block}
        
        Question: {query}
        
        Answer (use only the context above):"""

    def _is_answer_grounded(self, answer: str, contexts: List[str]) -> bool:
        """Check if the answer is grounded in the provided contexts."""
        # Check for empty or generic responses
        if not answer or answer.lower() in ["i don't know", "i don't know.", "", "i don't have enough information"]:
            return False
            
        # Check if the answer is too generic
        generic_indicators = [
            "as an ai", "i'm sorry", "i apologize", "i cannot", "i don't have", 
            "i don't know", "i do not know", "i'm not sure", "i am not sure"
        ]
        
        if any(indicator in answer.lower() for indicator in generic_indicators):
            return False
            
        # Check if any part of the answer appears in the contexts
        # Simple check: see if any non-common words from the answer are in the contexts
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        answer_words = set(word.lower() for word in answer.split() if word.lower() not in common_words and len(word) > 2)
        
        if not answer_words:
            return False
            
        context_text = " ".join(contexts).lower()
        matching_words = sum(1 for word in answer_words if word in context_text)
        
        # If at least 50% of the non-common words in the answer are in the context, consider it grounded
        return (matching_words / len(answer_words)) >= 0.5
