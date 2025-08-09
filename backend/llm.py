import re
import logging

logger = logging.getLogger(__name__)

def simple_answer_extraction(question, context):
    """Fallback method to extract answers directly from context"""
    
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Look for definition patterns
    if "what is" in question_lower or "define" in question_lower:
        # Look for definition patterns in context
        patterns = [
            r'definition:\s*([^•\n]+)',
            r'is\s+([^•\n\.]+)',
            r':\s+([A-Z][^•\n\.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                definition = match.group(1).strip()
                if len(definition) > 10:  # Ensure it's substantial
                    return f"Based on the provided materials: {definition}"
    
    # Look for direct answers
    sentences = re.split(r'[.!?]+', context)
    question_words = set(question_lower.split())
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
            
        sentence_words = set(sentence.lower().split())
        overlap = len(question_words.intersection(sentence_words))
        
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence.strip()
    
    if best_sentence and best_score >= 2:
        return f"Based on the provided materials: {best_sentence}"
    
    # If we have context but no good match, return first substantial sentence
    sentences = [s.strip() for s in re.split(r'[.!?]+', context) if len(s.strip()) > 20]
    if sentences:
        return f"Based on the provided materials: {sentences[0]}"
    
    return "I don't know based on the provided materials."

def generate_answer(question, context, max_tokens=256):
    """Try FLAN-T5 first, fallback to simple extraction"""
    
    context = context.strip()
    question = question.strip()
    
    logger.info(f"Generating answer for question: {question}")
    logger.info(f"Context preview: {context[:200]}...")
    
    if not context or context == "[NO TEXT]":
        return "I don't know based on the provided materials."
    
    # For now, let's use the simple extraction method since FLAN-T5 is having issues
    try:
        answer = simple_answer_extraction(question, context)
        logger.info(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I don't know based on the provided materials."