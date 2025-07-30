"""
Conversation Token Extractor Module

This module provides functionality to extract emotions, activities, and people
from conversation text using multiple strategies:
1. LLM-based extraction (using MLX/Mistral)
2. Rule-based extraction (using spaCy - optional fallback)
"""

import json
import re
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod


class TokenExtractor(ABC):
    """Abstract base class for token extraction strategies."""
    
    @abstractmethod
    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract emotions, activities, and people from text."""
        pass


class TogetherTokenExtractor(TokenExtractor):
    """Token extractor using Together API with LLM."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", api_key: str = None):
        """
        Initialize the Together API token extractor.
        
        Args:
            model_name: The Together model to use for extraction
            api_key: Together API key (if None, will look for TOGETHER_API_KEY env var)
        """
        try:
            from together import Together
            import os
            
            # Initialize Together client
            if api_key:
                self.client = Together(api_key=api_key)
            elif os.getenv('TOGETHER_API_KEY'):
                self.client = Together()
            else:
                print("Warning: No Together API key found. Please set TOGETHER_API_KEY environment variable or pass api_key parameter.")
                self.initialized = False
                return
                
            self.model_name = model_name
            self.initialized = True
        except ImportError:
            print("Warning: together package not available. Please install: pip install together")
            self.initialized = False
        except Exception as e:
            print(f"Warning: Failed to initialize Together client: {str(e)}")
            self.initialized = False
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract tokens using Together API analysis."""
        if not self.initialized:
            return {"emotions": [], "activities": [], "people": []}
        
        prompt = self._create_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
        
        # Debug: Uncomment to see raw response
            # print(f"DEBUG - Raw API Response:\n{response_text}\n")
            
            return self._parse_response(response_text)
        
        except Exception as e:
            print(f"Error calling Together API: {str(e)}")
            return {"emotions": [], "activities": [], "people": []}
    
    def _create_prompt(self, text: str) -> str:
        """Create the extraction prompt for the LLM."""
        return f"""You are a conversation analyzer. Extract emotions, activities, and people from the following conversation.

Rules:
- For EMOTIONS: Look for feelings and emotional states (happy, sad, angry, excited, frustrated, anxious, etc.)
- For ACTIVITIES: Extract ONLY the base verb forms. Examples:
  * "discussing" → discuss
  * "was running" → run
  * "decided to go" → decide, go
  * "planning to cook" → plan, cook
  * "couldn't finish" → finish
- For PEOPLE: Look for names and pronouns (John, Sarah, I, you, we, they, etc.)

You MUST respond in EXACTLY this format with square brackets:
EMOTIONS: [emotion1, emotion2, emotion3]
ACTIVITIES: [activity1, activity2, activity3]
PEOPLE: [person1, person2, person3]

If there are none for a category, use empty brackets: []

Conversation to analyze:
"{text}"

Now extract the tokens:"""
    
    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response into structured tokens."""
        tokens = {
            "emotions": [],
            "activities": [],
            "people": []
        }
        
        # Clean the response
        response = response.strip()
        
        # Try multiple parsing strategies
        
        # Strategy 1: Look for exact format with square brackets
        emotions_match = re.search(r'EMOTIONS?\s*:\s*\[(.*?)\]', response, re.IGNORECASE | re.DOTALL)
        if emotions_match:
            emotions_str = emotions_match.group(1).strip()
            if emotions_str:
                tokens["emotions"] = [e.strip().lower().strip('"\'') for e in emotions_str.split(',') if e.strip()]
        
        activities_match = re.search(r'ACTIVITIES?\s*:\s*\[(.*?)\]', response, re.IGNORECASE | re.DOTALL)
        if activities_match:
            activities_str = activities_match.group(1).strip()
            if activities_str:
                # Extract individual activities and clean them
                activities = []
                for a in activities_str.split(','):
                    activity = a.strip().lower().strip('"\'')
                    # Extract base verb from complex phrases
                    if activity:
                        # Handle common patterns like "couldn't finish" -> "finish"
                        activity = re.sub(r"^(couldn't|wouldn't|didn't|don't|won't|can't)\s+", "", activity)
                        activity = re.sub(r"^(was|were|been|being|is|are|am)\s+", "", activity)
                        activity = re.sub(r"^(decided to|planning to|going to|want to|need to)\s+", "", activity)
                        # Remove -ing endings to get base form
                        if activity.endswith('ing') and len(activity) > 4:
                            if activity.endswith('nning'):  # running -> run
                                activity = activity[:-4]
                            elif activity.endswith('tting'):  # getting -> get
                                activity = activity[:-4]
                            else:
                                activity = activity[:-3]
                        activities.append(activity)
                tokens["activities"] = activities
        
        people_match = re.search(r'PEOPLE?\s*:\s*\[(.*?)\]', response, re.IGNORECASE | re.DOTALL)
        if people_match:
            people_str = people_match.group(1).strip()
            if people_str:
                people_list = []
                for p in people_str.split(','):
                    p = p.strip().strip('"\'')
                    if p:
                        # Keep pronouns lowercase, names as-is
                        if p.lower() in ['i', 'you', 'we', 'they', 'he', 'she', 'me', 'us', 'him', 'her']:
                            people_list.append(p.lower())
                        else:
                            people_list.append(p)
                tokens["people"] = people_list
        
        # Strategy 2: If square brackets not found, try with different patterns
        if not emotions_match:
            # Try pattern without brackets
            emotions_match2 = re.search(r'EMOTIONS?\s*:\s*([^\n]+)', response, re.IGNORECASE)
            if emotions_match2:
                emotions_str = emotions_match2.group(1).strip()
                # Remove any trailing text after the list
                emotions_str = re.sub(r'\s*(ACTIVITIES?|PEOPLE?|$).*', '', emotions_str, flags=re.IGNORECASE | re.DOTALL)
                if emotions_str and emotions_str.lower() not in ['none', 'n/a', '[]', 'empty']:
                    tokens["emotions"] = [e.strip().lower().strip('"\'[]') for e in re.split(r'[,;]', emotions_str) if e.strip()]
        
        if not activities_match:
            activities_match2 = re.search(r'ACTIVITIES?\s*:\s*([^\n]+)', response, re.IGNORECASE)
            if activities_match2:
                activities_str = activities_match2.group(1).strip()
                activities_str = re.sub(r'\s*(EMOTIONS?|PEOPLE?|$).*', '', activities_str, flags=re.IGNORECASE | re.DOTALL)
                if activities_str and activities_str.lower() not in ['none', 'n/a', '[]', 'empty']:
                    activities = []
                    for a in re.split(r'[,;]', activities_str):
                        activity = a.strip().lower().strip('"\'[]')
                        if activity:
                            # Same processing as above
                            activity = re.sub(r"^(couldn't|wouldn't|didn't|don't|won't|can't)\s+", "", activity)
                            activity = re.sub(r"^(was|were|been|being|is|are|am)\s+", "", activity)
                            activity = re.sub(r"^(decided to|planning to|going to|want to|need to)\s+", "", activity)
                            if activity.endswith('ing') and len(activity) > 4:
                                if activity.endswith('nning'):
                                    activity = activity[:-4]
                                elif activity.endswith('tting'):
                                    activity = activity[:-4]
                                else:
                                    activity = activity[:-3]
                            activities.append(activity)
                    tokens["activities"] = activities
        
        if not people_match:
            people_match2 = re.search(r'PEOPLE?\s*:\s*([^\n]+)', response, re.IGNORECASE)
            if people_match2:
                people_str = people_match2.group(1).strip()
                people_str = re.sub(r'\s*(EMOTIONS?|ACTIVITIES?|$).*', '', people_str, flags=re.IGNORECASE | re.DOTALL)
                if people_str and people_str.lower() not in ['none', 'n/a', '[]', 'empty']:
                    people_list = []
                    for p in re.split(r'[,;]', people_str):
                        p = p.strip().strip('"\'[]')
                        if p:
                            if p.lower() in ['i', 'you', 'we', 'they', 'he', 'she', 'me', 'us', 'him', 'her']:
                                people_list.append(p.lower())
                            else:
                                people_list.append(p)
                    tokens["people"] = people_list
        
        # Clean up any duplicates and empty strings
        tokens["emotions"] = list(dict.fromkeys([e for e in tokens["emotions"] if e]))
        tokens["activities"] = list(dict.fromkeys([a for a in tokens["activities"] if a]))
        tokens["people"] = list(dict.fromkeys([p for p in tokens["people"] if p]))
        
        return tokens


class RuleBasedTokenExtractor(TokenExtractor):
    """Rule-based token extractor using keyword matching and NLP patterns."""
    
    def __init__(self):
        """Initialize the rule-based extractor."""
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'joyful', 'pleased', 'delighted', 'cheerful', 'glad', 'excited'],
            'sad': ['sad', 'unhappy', 'depressed', 'miserable', 'sorrowful', 'upset', 'disappointed'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'outraged'],
            'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'tense', 'uneasy', 'concerned'],
            'content': ['content', 'satisfied', 'peaceful', 'calm', 'relaxed', 'comfortable'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered'],
            'confident': ['confident', 'sure', 'certain', 'assured', 'positive', 'determined']
        }
        
        # Common activity verbs
        self.activity_verbs = {
            'work', 'study', 'run', 'walk', 'eat', 'drink', 'sleep', 'talk', 'discuss',
            'plan', 'think', 'create', 'build', 'write', 'read', 'watch', 'listen',
            'cook', 'clean', 'drive', 'travel', 'meet', 'call', 'text', 'email',
            'finish', 'start', 'go', 'come', 'decide', 'prepare', 'organize'
        }
        
        # Common pronouns and name patterns
        self.pronouns = ['i', 'you', 'we', 'they', 'he', 'she', 'me', 'us', 'him', 'her']
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract tokens using rule-based patterns."""
        text_lower = text.lower()
        
        # Extract emotions
        emotions = []
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                emotions.append(emotion)
        
        # Extract activities (enhanced verb matching)
        activities = []
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            # Check base forms and common variations
            if word in self.activity_verbs:
                activities.append(word)
            elif word.endswith('ing') and len(word) > 4:
                # Handle -ing endings
                if word.endswith('nning'):  # running -> run
                    base = word[:-4]
                elif word.endswith('tting'):  # getting -> get
                    base = word[:-4]
                else:
                    base = word[:-3]
                if base in self.activity_verbs:
                    activities.append(base)
            elif word.endswith('ed') and len(word) > 3:
                # Handle -ed endings
                base = word[:-2] if word.endswith('ed') else word[:-1]
                if base in self.activity_verbs:
                    activities.append(base)
        
        # Extract people (names and pronouns)
        people = []
        # Extract capitalized words that might be names (but exclude sentence starters)
        words_in_text = text.split()
        for i, word in enumerate(words_in_text):
            if re.match(r'^[A-Z][a-z]+$', word):
                # Skip if it's the first word of a sentence (likely not a name)
                if i == 0 or words_in_text[i-1].endswith('.'):
                    continue
                people.append(word)
        
        # Add pronouns
        for pronoun in self.pronouns:
            if re.search(rf'\b{pronoun}\b', text_lower):
                people.append(pronoun)
        
        return {
            "emotions": list(dict.fromkeys(emotions)),  # Remove duplicates while preserving order
            "activities": list(dict.fromkeys(activities)),
            "people": list(dict.fromkeys(people))
        }


class MixedTokenExtractor(TokenExtractor):
    """Mixed token extractor that combines rule-based and API methods for enhanced accuracy."""
    
    def __init__(self, api_key: str = None, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Initialize the mixed extractor.
        
        Args:
            api_key: Together API key (optional)
            model_name: Model to use with Together API
        """
        # Initialize both extractors
        self.rule_extractor = RuleBasedTokenExtractor()
        self.api_extractor = TogetherTokenExtractor(model_name=model_name, api_key=api_key)
        
        # Track which extractors are available
        self.has_api = self.api_extractor.initialized
        self.has_rules = True  # Rule-based is always available
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract tokens using both rule-based and API methods, then combine results."""
        # Get results from both methods
        rule_results = self.rule_extractor.extract(text)
        
        if self.has_api:
            api_results = self.api_extractor.extract(text)
        else:
            # Fallback to rule-based only if API is unavailable
            return rule_results
        
        # Combine results intelligently
        combined_results = self._combine_results(rule_results, api_results, text)
        
        return combined_results
    
    def _combine_results(self, rule_results: Dict[str, List[str]], 
                        api_results: Dict[str, List[str]], 
                        text: str) -> Dict[str, List[str]]:
        """
        Intelligently combine results from both extractors.
        
        Strategy:
        - Use API results as primary (more contextual)
        - Add rule-based results that API might have missed
        - Remove duplicates and filter false positives
        """
        combined = {
            "emotions": [],
            "activities": [],
            "people": []
        }
        
        text_lower = text.lower()
        
        # Combine emotions (API primary, rules supplement)
        combined["emotions"] = list(api_results["emotions"])
        for emotion in rule_results["emotions"]:
            if emotion not in combined["emotions"]:
                # Add rule-based emotion if it's strongly supported by keywords
                if self._validate_emotion_in_text(emotion, text_lower):
                    combined["emotions"].append(emotion)
        
        # Combine activities (merge both, API gets priority for complex forms)
        combined["activities"] = list(api_results["activities"])
        for activity in rule_results["activities"]:
            if activity not in combined["activities"]:
                combined["activities"].append(activity)
        
        # Combine people (merge both, prioritize pronouns from rules, names from API)
        # Start with API results (better at context and names)
        combined["people"] = list(api_results["people"])
        
        # Add pronouns from rule-based (more reliable for pronouns)
        pronouns = ['i', 'you', 'we', 'they', 'he', 'she', 'me', 'us', 'him', 'her']
        for person in rule_results["people"]:
            if person.lower() in pronouns and person not in combined["people"]:
                combined["people"].append(person)
            elif person not in combined["people"] and len(person) > 2:
                # Add names that rule-based found but API missed
                combined["people"].append(person)
        
        # Clean up and remove duplicates while preserving order
        for key in combined:
            combined[key] = list(dict.fromkeys([item for item in combined[key] if item.strip()]))
        
        return combined
    
    def _validate_emotion_in_text(self, emotion: str, text_lower: str) -> bool:
        """Validate if an emotion detected by rules is actually supported in the text."""
        emotion_keywords = {
            'happy': ['happy', 'joy', 'joyful', 'pleased', 'delighted', 'cheerful', 'glad'],
            'sad': ['sad', 'unhappy', 'depressed', 'miserable', 'sorrowful', 'upset'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
            'excited': ['excited', 'thrilled', 'enthusiastic', 'eager', 'pumped'],
            'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'tense', 'uneasy'],
            'content': ['content', 'satisfied', 'peaceful', 'calm', 'relaxed'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'confident': ['confident', 'sure', 'certain', 'assured', 'positive']
        }
        
        if emotion in emotion_keywords:
            return any(keyword in text_lower for keyword in emotion_keywords[emotion][:3])  # Check top 3 keywords
        return False


class ConversationAnalyzer:
    """Main class for analyzing conversations and extracting tokens."""
    
    def __init__(self, strategy: str = "mixed", api_key: str = None, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Initialize the conversation analyzer.
        
        Args:
            strategy: Extraction strategy to use ("together", "rule_based", or "mixed")
            api_key: Together API key (optional, required for "together" and "mixed")
            model_name: Model to use with Together API
        """
        if strategy == "together":
            self.extractor = TogetherTokenExtractor(model_name=model_name, api_key=api_key)
        elif strategy == "rule_based":
            self.extractor = RuleBasedTokenExtractor()
        elif strategy == "mixed":
            self.extractor = MixedTokenExtractor(api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Supported strategies: 'together', 'rule_based', 'mixed'")
    
    def analyze(self, conversation: Union[str, List[str]]) -> Dict[str, List[str]]:
        """
        Analyze a conversation and extract tokens.
        
        Args:
            conversation: Single conversation string or list of utterances
            
        Returns:
            Dictionary containing emotions, activities, and people
        """
        if isinstance(conversation, list):
            conversation = " ".join(conversation)
        
        return self.extractor.extract(conversation)
    
    def analyze_batch(self, conversations: List[str]) -> List[Dict[str, List[str]]]:
        """
        Analyze multiple conversations.
        
        Args:
            conversations: List of conversation texts
            
        Returns:
            List of extraction results
        """
        return [self.analyze(conv) for conv in conversations]
    
    def format_output(self, tokens: Dict[str, List[str]], format_type: str = "json") -> str:
        """
        Format the extracted tokens for output.
        
        Args:
            tokens: Extracted tokens dictionary
            format_type: Output format ("json", "text", or "summary")
            
        Returns:
            Formatted string
        """
        if format_type == "json":
            return json.dumps(tokens, indent=2)
        elif format_type == "text":
            lines = []
            lines.append(f"Emotions: {', '.join(tokens['emotions']) or 'None'}")
            lines.append(f"Activities: {', '.join(tokens['activities']) or 'None'}")
            lines.append(f"People: {', '.join(tokens['people']) or 'None'}")
            return "\n".join(lines)
        elif format_type == "summary":
            total = len(tokens['emotions']) + len(tokens['activities']) + len(tokens['people'])
            return f"Found {len(tokens['emotions'])} emotions, {len(tokens['activities'])} activities, and {len(tokens['people'])} people (Total: {total} tokens)"
        else:
            raise ValueError(f"Unknown format type: {format_type}")


# Convenience functions
def extract_tokens(text: str, strategy: str = "mixed", api_key: str = None) -> Dict[str, List[str]]:
    """
    Quick function to extract tokens from text.
    
    Args:
        text: Conversation text to analyze
        strategy: Extraction strategy ("together", "rule_based", or "mixed")
        api_key: Together API key (optional, required for "together" and "mixed")
        
    Returns:
        Dictionary with emotions, activities, and people
    """
    analyzer = ConversationAnalyzer(strategy=strategy, api_key=api_key)
    return analyzer.analyze(text)


def extract_from_audio(audio_path: str, strategy: str = "mixed", api_key: str = None) -> Dict[str, List[str]]:
    """
    Extract tokens from audio file (requires speech-to-text).
    
    Args:
        audio_path: Path to audio file
        strategy: Extraction strategy
        api_key: Together API key (optional)
        
    Returns:
        Dictionary with emotions, activities, and people
    """
    # Placeholder for audio transcription
    # In production, you would use Whisper or another STT service
    raise NotImplementedError("Audio transcription not yet implemented. Use extract_tokens() with transcribed text.")


if __name__ == "__main__":
    # Demo usage
    test_conversations = [
        "I'm really excited about the project! John and I were discussing it yesterday while having coffee.",
        "Sarah felt frustrated when she couldn't finish her work. She decided to go for a run to clear her mind.",
        "We're planning to cook dinner together. You seem happy about it!",
    ]
    
    print("Testing all extraction strategies:")
    print("=" * 60)
    
    # Test all three strategies
    strategies = ["rule_based", "together", "mixed"]
    
    for strategy in strategies:
        print(f"\n🔍 Testing {strategy.upper().replace('_', ' ')} Strategy:")
        print("-" * 40)
        
        try:
            analyzer = ConversationAnalyzer(strategy=strategy)
            
            # Test with first conversation
            test_text = test_conversations[0]
            print(f"Text: {test_text}")
            
            tokens = analyzer.analyze(test_text)
            print(analyzer.format_output(tokens, "text"))
            
        except Exception as e:
            print(f"❌ Error with {strategy} strategy: {str(e)}")
        
        print() 