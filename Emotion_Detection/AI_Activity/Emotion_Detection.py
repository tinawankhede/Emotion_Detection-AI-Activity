from transformers import pipeline

# Load pre-trained emotion detection model
emotion_classifier = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

def identify_emotions(text):
    # Get predictions from the model
    results = emotion_classifier(text)
    
    # Organize and display the results
    emotions = {res['label']: res['score'] for res in results[0]}
    return emotions

# Example usage
if __name__ == "__main__":
    input_text = "I feel so sad and disappointed."
    emotions = identify_emotions(input_text)
    
    print("Emotion scores:")
    for emotion, score in emotions.items():
        print(f"{emotion}: {score:.4f}")
