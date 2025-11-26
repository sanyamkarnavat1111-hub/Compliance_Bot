from logger_config import get_logger

logger = get_logger(__file__)

class ArabicEnglishDetector:
    def __init__(self):
        try:
            # Arabic character range in Unicode: \u0600 to \u06FF
            self.arabic_range = range(0x0600, 0x0700)
            # Common English characters (ASCII + common punctuation)
            self.english_range = range(0x20, 0x7F)
        except Exception as e:
            logger.info(f"Error initializing ArabicEnglishDetector: {e}")
            raise

    def _is_arabic_char(self, char):
        """Check if a character is Arabic."""
        try:
            return ord(char) in self.arabic_range
        except Exception as e:
            logger.info(f"Error checking Arabic character: {e}")
            return False

    def _is_english_char(self, char):
        """Check if a character is English (ASCII)."""
        try:
            return ord(char) in self.english_range
        except Exception as e:
            logger.info(f"Error checking English character: {e}")
            return False

    def detect_language(self, input_text, block_size=10000):
        """
        Detect Arabic and English content in a large text.
        
        Args:
            input_text (str): The text to analyze
            block_size (int): Process text in blocks of this size
            
        Returns:
            str: 'arabic' or 'english'
        """
        try:
            if not isinstance(input_text, str):
                logger.info("Warning: Input must be a string, returning 'english' as default")
                return 'english'
                
            total_chars = len(input_text)
            if total_chars == 0:
                return 'english'  # Default to english for empty string

            language_counts = {'arabic': 0, 'english': 0}
            
            # Process in blocks to handle very large strings
            for start in range(0, total_chars, block_size):
                try:
                    block = input_text[start:start + block_size]
                    
                    # Count characters in each block
                    for char in block:
                        try:
                            if self._is_arabic_char(char):
                                language_counts['arabic'] += 1
                            elif self._is_english_char(char):
                                language_counts['english'] += 1
                        except Exception as char_error:
                            logger.info(f"Error processing character: {char_error}")
                            continue
                except Exception as block_error:
                    logger.info(f"Error processing block starting at {start}: {block_error}")
                    continue
            
            total_chars = language_counts['arabic'] + language_counts['english']
            if total_chars == 0:
                return 'english'  # Default to english if no characters matched
                
            arabic_ratio = language_counts['arabic'] / total_chars
            return 'arabic' if arabic_ratio > 0.3 else 'english'
            
        except Exception as e:
            logger.info(f"Error in language detection: {e}")
            return 'english'  # Default to english in case of error

if __name__ == "__main__":
    try:
        arabic_english_detector = ArabicEnglishDetector()
        
        # Example 1 - Empty string
        result = arabic_english_detector.detect_language("")
        logger.info(result)
    except Exception as e:
        logger.info(f"Error in main: {e}")
        exit(1)