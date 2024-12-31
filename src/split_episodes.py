import re
import os

def split_episodes(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split content into episodes - handling all possible formats
    episodes = re.split(r'(?=[#]{1,2} (?:Ep\.|Episode) \d+[^\n]*)', content)
    
    # Remove any empty episodes
    episodes = [ep for ep in episodes if ep.strip()]
    
    # Track processed episode numbers to avoid duplicates
    processed_episodes = set()
    
    # Save each episode to a separate file
    for episode in episodes:
        # Extract episode number - handle all formats
        match = re.search(r'[#]{1,2} (?:Ep\.|Episode) (\d+)', episode)
        if match:
            episode_num = match.group(1)
            # Skip if we've already processed this episode
            if episode_num in processed_episodes:
                continue
                
            processed_episodes.add(episode_num)
            
            # Pad episode number with zeros
            padded_num = episode_num.zfill(2)
            
            # Create filename
            filename = f"Episode_{padded_num}.md"
            filepath = os.path.join(output_dir, filename)
            
            # Write episode content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(episode.strip())
            print(f"Created {filename}")

if __name__ == "__main__":
    input_file = "Relevance Realization - Awakening from the Meaning Crisis - Notes and Transcriptions.md"
    output_dir = "50 Episodes in Relevance Realization"
    split_episodes(input_file, output_dir) 