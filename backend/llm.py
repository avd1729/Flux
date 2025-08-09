from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_answer_with_flan(question, context, max_tokens=128):
    
    prompt = f"Please answer this question using only the provided information.\n\nInformation:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        # Tokenize with proper truncation
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )
        
        # Generate with good parameters for FLAN-T5
        with tokenizer.as_target_tokenizer():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_tokens,
                num_beams=2,  # Small beam search for better quality
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # Avoid repetitive text
            )
        
        # Decode the response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"FLAN-T5 raw output: {generated_text}")
        
        # Clean up the response
        answer = generated_text.strip()
        
        # If the answer is too short, empty, or just repeats the prompt
        if (not answer or 
            len(answer) < 5 or 
            answer.lower() in ["i don't know", "i don't know.", "no", "yes"] or
            prompt.lower() in answer.lower()):
            logger.info("FLAN-T5 gave insufficient answer, using fallback")
            return None
            
        # Format the answer nicely
        if not answer.startswith("Based on"):
            answer = f"Based on the provided materials: {answer}"
            
        return answer
        
    except Exception as e:
        logger.error(f"FLAN-T5 generation failed: {e}")
        return None

def generate_answer(question, context, max_tokens=256):
    
    context = context.strip()
    question = question.strip()
    
    logger.info(f"Generating answer for question: {question}")
    logger.info(f"Context length: {len(context)} chars")
    
    if not context or context == "[NO TEXT]":
        return "I don't know based on the provided materials."

    logger.info("Attempting FLAN-T5 generation...")
    flan_answer = generate_answer_with_flan(question, context, max_tokens=128)
    
    if flan_answer:
        logger.info(f"FLAN-T5 succeeded: {flan_answer[:100]}...")
        return flan_answer
    
    return "I don't know based on the provided materials."