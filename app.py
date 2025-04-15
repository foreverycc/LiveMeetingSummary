import streamlit as st
import whisper
import pyaudio
import numpy as np
import threading
import time
import json
import google.generativeai as genai
import openai
import queue
import os
from datetime import datetime
import torch
from typing import Optional, Dict, List, Any
import difflib  # Add this to your imports

# Configure page settings
st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summaries' not in st.session_state:
    st.session_state.summaries = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'last_summary_time' not in st.session_state:
    st.session_state.last_summary_time = time.time()
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'questions_answers' not in st.session_state:
    st.session_state.questions_answers = []
# Add a queue for transcribed text results
if 'transcript_q' not in st.session_state:
    st.session_state.transcript_q = queue.Queue()
if 'needs_update' not in st.session_state:
    st.session_state.needs_update = False
if 'last_correction' not in st.session_state:
    st.session_state.last_correction = None
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ""
if 'last_summary_update' not in st.session_state:
    st.session_state.last_summary_update = None
if 'highlighted_summary' not in st.session_state:
    st.session_state.highlighted_summary = ""

# LLM provider configuration
@st.cache_resource
def load_api_keys():
    """Load API keys from environment variables or Streamlit secrets."""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GENAI_API_KEY": os.getenv("GENAI_API_KEY")
    }
    
    # Initialize API clients
    if keys["GENAI_API_KEY"]:
        genai.configure(api_key=keys["GENAI_API_KEY"])
    if keys["OPENAI_API_KEY"]:
        openai.api_key = keys["OPENAI_API_KEY"]
        
    return keys

api_keys = load_api_keys()

# Load Whisper model
@st.cache_resource
def load_whisper_model(model_name: str):
    """Load and return the specified Whisper model."""
    if model_name == "None":
        return None
    return whisper.load_model(model_name)

# Audio recording functions
def setup_audio_stream(sample_rate=16000, chunk_size=1024):
    """Setup and return audio stream configuration."""
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )
    return audio, stream, sample_rate, chunk_size

def record_audio(stop_event, audio_q):
    """Record audio and add to queue until stop event is set."""
    try:
        audio, stream, sample_rate, chunk_size = setup_audio_stream()
        
        while not stop_event.is_set():
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            audio_q.put(audio_data)
            
        # Clean up resources
        stream.stop_stream()
        stream.close()
        audio.terminate()
    except Exception as e:
        print(f"Error during audio recording: {str(e)}")

def process_audio(stop_event, audio_q, transcript_q, whisper_model_instance):
    """Process audio with overlapping windows to enable correction of previous text."""
    buffer = []
    previous_segments = np.array([])  # Initialize as empty NumPy array
    overlap_seconds = 2  # Amount of audio to overlap with previous processing
    sample_rate = 16000  # Whisper's expected sample rate
    overlap_samples = int(overlap_seconds * sample_rate)
    
    if not whisper_model_instance:
        print("Whisper model not loaded, processing thread exiting.")
        return
    
    while not stop_event.is_set() or not audio_q.empty():
        # Get audio chunk with a short timeout
        try:
            audio_chunk = audio_q.get(timeout=0.1)
            buffer.append(audio_chunk)
            
            # Process when buffer gets large enough (about 2-3 seconds of audio)
            if len(buffer) >= 30:  # Adjust based on your chunk size
                # Create processing segment with overlap from previous audio
                current_audio = np.concatenate(buffer)
                
                # Include overlap from previous segments if available
                if len(previous_segments) > 0:  # Fix: Use len() instead of direct boolean eval
                    # Combine last portion of previous segments with current audio
                    overlap_audio = previous_segments[-overlap_samples:] if len(previous_segments) > overlap_samples else previous_segments
                    process_audio = np.concatenate([overlap_audio, current_audio])
                else:
                    process_audio = current_audio
                
                # Update previous segments for next processing
                previous_segments = np.concatenate([previous_segments, current_audio]) if len(previous_segments) > 0 else current_audio
                # Trim previous_segments to prevent excessive memory usage
                max_history = sample_rate * 10  # 10 seconds of history
                if len(previous_segments) > max_history:
                    previous_segments = previous_segments[-max_history:]
                
                # Transcribe with context
                result = whisper_model_instance.transcribe(process_audio, fp16=torch.cuda.is_available())
                transcribed_text = result["text"].strip()
                
                if transcribed_text:
                    # Send both the text and position information
                    transcript_q.put({
                        "text": transcribed_text,
                        "is_correction": len(previous_segments) > overlap_samples,
                        "overlap_seconds": overlap_seconds
                    })
                
                # Clear current buffer but keep overlap for next processing
                buffer = []
                
        except queue.Empty:
            if not stop_event.is_set():
                time.sleep(0.1)
            else:
                break

def generate_summary():
    """Generate a completely fresh summary of the current transcript and highlight changes."""
    if not st.session_state.transcript.strip():
        return
    
    try:
        # Store the previous summary before generating a new one
        previous_summary = st.session_state.current_summary
        
        # Create a prompt that encourages a fresh summary each time
        prompt = f"""Create a concise summary of the following meeting transcript in Markdown format.
        
Transcript:
{st.session_state.transcript}

Previous Summary:
{previous_summary}

Guidelines:
1. Create a fresh summary based on the transcript and the previous summaries, make minimal changes to the previous summary.
2. Focus only on the key information and main points.
3. Use the following Markdown formatting:
   - ## Meeting Overview (as the main heading)
   - Bullet points for key discussion items
   - **Bold text** for important decisions or action items
   - Brief section at the end for "Next Steps" if applicable

Keep the summary focused and concise. Remove any redundant information.

Summary:"""

        # Use selected LLM to generate summary
        if st.session_state.llm_provider == "Gemini":
            model = genai.GenerativeModel(st.session_state.llm_model)
            response = model.generate_content(prompt, generation_config={"temperature": 0.1})
            new_summary = response.text
        elif st.session_state.llm_provider == "OpenAI":
            response = openai.ChatCompletion.create(
                model=st.session_state.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            new_summary = response.choices[0].message.content
        
        # Store the raw new summary
        st.session_state.current_summary = new_summary
        
        # Compare with previous summary and store highlighted version
        st.session_state.highlighted_summary = detect_summary_changes(previous_summary, new_summary)
        
        # Store timestamp of when the summary was last updated
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.last_summary_timestamp = timestamp
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")

def answer_question(question: str):
    """Answer a question based on the transcript using the selected LLM."""
    if not st.session_state.transcript.strip():
        return "No transcript available to answer questions."
    
    try:
        prompt = f"""Based on the following meeting transcript, please answer this question:
        
Question: {question}

Meeting Transcript:
{st.session_state.transcript}

Answer:"""

        answer = ""
        # Use selected LLM to generate answer
        if st.session_state.llm_provider == "Gemini":
            model = genai.GenerativeModel(st.session_state.llm_model)
            response = model.generate_content(prompt)
            answer = response.text
        elif st.session_state.llm_provider == "OpenAI":
            response = openai.ChatCompletion.create(
                model=st.session_state.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            answer = response.choices[0].message.content
        
        # Store the Q&A pair
        if answer:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.questions_answers.append({
                "timestamp": timestamp,
                "question": question,
                "answer": answer
            })
            return answer
        return "Unable to generate an answer."
    except Exception as e:
        return f"Error answering question: {str(e)}"

def export_session_data():
    """Export all session data to a JSON file."""
    session_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transcript": st.session_state.transcript,
        "summary": st.session_state.current_summary,
        "questions_answers": st.session_state.questions_answers
    }
    
    return json.dumps(session_data, indent=2)

def start_recording():
    """Start recording and processing audio."""
    if st.session_state.recording:
        return
    
    # Clear previous session data if requested
    if st.session_state.clear_previous:
        st.session_state.transcript = ""
        st.session_state.current_summary = ""  # Clear current summary
        st.session_state.last_summary_update = None
        st.session_state.questions_answers = []
        # Ensure queue is cleared as well if resetting
        while not st.session_state.audio_queue.empty():
            try:
                st.session_state.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    # Set up threading events
    stop_event = threading.Event()
    st.session_state.stop_event = stop_event
    st.session_state.last_summary_time = time.time()
    
    # Load selected Whisper model
    whisper_selection = st.session_state.whisper_selection
    if whisper_selection != "None":
        st.session_state.whisper_model = load_whisper_model(whisper_selection)
    else:
        st.session_state.whisper_model = None
    
    # Fetch the queue object from session state
    # The initialization at the top ensures it exists here
    audio_q = st.session_state.audio_queue
    # Fetch the new transcript queue
    transcript_q = st.session_state.transcript_q
    
    # Start recording and processing threads, passing the queue object
    # Ensure whisper_model is loaded before accessing it
    if st.session_state.whisper_model:
        # Pass audio_q as an argument
        rec_thread = threading.Thread(target=record_audio, args=(stop_event, audio_q))
        # Pass audio_q, transcript_q, and the loaded whisper_model instance
        proc_thread = threading.Thread(
            target=process_audio,
            args=(stop_event, audio_q, transcript_q, st.session_state.whisper_model)
        )

        rec_thread.start()
        proc_thread.start()
        
        st.session_state.recording = True
        st.session_state.rec_thread = rec_thread
        st.session_state.proc_thread = proc_thread

def stop_recording():
    """Stop recording and processing audio."""
    if not st.session_state.recording:
        return
    
    # Signal threads to stop
    st.session_state.stop_event.set()
    
    # Wait for threads to finish
    try:
        st.session_state.rec_thread.join(timeout=5) # Add timeout
        st.session_state.proc_thread.join(timeout=5) # Add timeout
    except Exception as e:
        st.warning(f"Error joining threads: {e}") # Log potential issues joining

    # Process any remaining audio in the queue after stopping recording
    # Use the queue directly from session state here as we are in the main thread context
    try:
        # Pass the queue from session state to process any final chunks
        # Need a way to call process_audio for final processing safely.
        # Simpler: Generate final summary based on current transcript.
        pass # Let the proc_thread finish processing via the join() call above.
    except Exception as e:
         st.warning(f"Error processing remaining audio: {e}")

    # Generate a final summary
    generate_summary()

    st.session_state.recording = False
    # Clean up references (optional but good practice)
    if 'rec_thread' in st.session_state: del st.session_state['rec_thread']
    if 'proc_thread' in st.session_state: del st.session_state['proc_thread']
    if 'stop_event' in st.session_state: del st.session_state['stop_event']

def find_common_suffix(text1, text2, max_words=10):
    """Find common text between the end of text1 and anywhere in text2."""
    # Handle empty strings
    if not text1 or not text2:
        return None
        
    words1 = text1.split()[-max_words:] if len(text1.split()) > max_words else text1.split()
    
    # Try successively smaller suffixes of text1
    for i in range(len(words1)):
        suffix = " ".join(words1[i:])
        if len(suffix) > 10 and suffix in text2:  # Ensure suffix is substantial
            return suffix
            
    # If no long match found, try matching last few words exactly
    if len(words1) >= 3:
        short_suffix = " ".join(words1[-3:])
        if short_suffix in text2:
            return short_suffix
            
    return None

def find_new_content(current_text, new_text, overlap_threshold=0.7):
    """
    Identify only the truly new content from new_text compared to current_text.
    Returns tuple: (new content to append, whether it's a correction)
    """
    # If current text is empty, everything is new
    if not current_text:
        return new_text, False
        
    # Split into sentences for better comparison
    import re
    current_sentences = re.split(r'(?<=[.!?])\s+', current_text)
    new_sentences = re.split(r'(?<=[.!?])\s+', new_text)
    
    # Check if the new text is completely contained in the current text
    if new_text in current_text:
        return "", False
    
    # Find the first unique sentence in the new text
    for i, new_sentence in enumerate(new_sentences):
        # Skip very short sentences as they might match commonly
        if len(new_sentence.split()) < 3:
            continue
            
        # Check if this sentence exists in current text
        if new_sentence in current_text:
            continue
            
        # If we get here, we've found the first unique sentence
        unique_start_idx = i
        break
    else:
        # If all sentences exist in current text
        return "", False
    
    # Check if this is a correction (significant overlap at start)
    # or just new content to append
    
    # First, check if the beginning of new_text matches the end of current_text
    common_suffix = find_common_suffix(current_text, new_text, max_words=15)
    if common_suffix and len(common_suffix) > 10:
        # This is likely a correction
        correction_point = current_text.rfind(common_suffix)
        if correction_point > 0:
            # Return the entire new text as a correction
            return new_text, True
    
    # Otherwise, return only the new sentences
    return " ".join(new_sentences[unique_start_idx:]), False

# Add to the top of your app.py file
st.markdown("""
<style>
.correction {
    background-color: rgba(255, 255, 0, 0.2);
    animation: highlight 2s ease-out;
}
@keyframes highlight {
    from {background-color: rgba(255, 255, 0, 0.4);}
    to {background-color: rgba(255, 255, 0, 0);}
}
</style>
""", unsafe_allow_html=True)

# Then modify transcript display to use HTML when needed
def format_transcript_with_highlight(transcript, last_updated_portion=None):
    """Format transcript with visual highlighting for recently corrected text."""
    if not transcript:
        return ""
        
    # Escape any HTML characters to prevent rendering issues
    # (This is important to avoid breaking the markdown parser)
    import html
    safe_transcript = html.escape(transcript)
    
    if last_updated_portion and last_updated_portion in transcript:
        # Escape the portion to highlight too
        safe_highlight = html.escape(last_updated_portion)
        # Highlight the corrected portion
        highlighted = safe_transcript.replace(
            safe_highlight, 
            f'<span class="correction">{safe_highlight}</span>', 
            1
        )
        return highlighted
    
    return safe_transcript

def format_summary_with_highlight(summary, previous_summary=None):
    """Format summary with visual highlighting for updated content."""
    if not summary:
        return ""
    
    # Escape any HTML characters to prevent rendering issues
    import html
    safe_summary = html.escape(summary)
    
    # No highlighting needed if there's no previous summary
    if not previous_summary:
        return safe_summary
    
    # For simplicity, we'll highlight the entire summary when it's updated
    # A more sophisticated approach could try to identify specific changed sections
    return f'<div class="correction">{safe_summary}</div>'

# Add to the start of the file or where appropriate
def detect_summary_changes(old_summary, new_summary):
    """
    Compare old and new summaries to identify and highlight the changed portions.
    Returns the new summary with HTML highlighting for changed parts.
    """
    if not old_summary:
        return new_summary  # No highlighting needed for the first summary
    
    # Split summaries into lines for better comparison
    old_lines = old_summary.splitlines()
    new_lines = new_summary.splitlines()
    
    # Use difflib to compare the two summaries line by line
    differ = difflib.Differ()
    diff = list(differ.compare(old_lines, new_lines))
    
    # Process the diff results to identify added/changed lines
    highlighted_lines = []
    for line in diff:
        if line.startswith('+ '):  # Added line
            # Extract the content without the diff marker
            content = line[2:]
            highlighted_lines.append(f'<span class="correction">{content}</span>')
        elif line.startswith('- '):  # Removed line - we skip these
            continue
        elif line.startswith('  '):  # Unchanged line
            # Extract the content without the diff marker
            content = line[2:]
            highlighted_lines.append(content)
        elif line.startswith('? '):  # Diff information line - we skip these
            continue
    
    # Join the highlighted lines back together
    return '<br>'.join(highlighted_lines)

# UI Layout
st.title("🎙️ AI Meeting Assistant")

# Create a sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Whisper model selection
    whisper_models = ["None", "tiny", "base", "small", "medium", "large"]
    st.session_state.whisper_selection = st.selectbox(
        "Select Whisper Model", 
        whisper_models,
        index=1
    )
    
    # LLM provider and model selection
    st.session_state.llm_provider = st.selectbox(
        "LLM Provider",
        ["Gemini", "OpenAI"],
        index=0
    )
    
    if st.session_state.llm_provider == "Gemini":
        st.session_state.llm_model = st.selectbox(
            "Gemini Model",
            ["gemini-1.0-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
            index=2
        )
    else:
        st.session_state.llm_model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4o"],
            index=0
        )
    
    # Summary update frequency
    st.session_state.summary_frequency = st.slider(
        "Summary Update Frequency (seconds)",
        min_value=5,
        max_value=30,
        value=10,
        step=1
    )
    
    # Option to clear previous data
    st.session_state.clear_previous = st.checkbox("Clear previous data when starting", value=True)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording", type="primary", disabled=st.session_state.recording):
            start_recording()
    
    with col2:
        if st.button("Stop Recording", type="secondary", disabled=not st.session_state.recording):
            stop_recording()
    
    # Export option
    if st.session_state.transcript:
        session_data = export_session_data()
        st.download_button(
            label="Download Session Data",
            data=session_data,
            file_name=f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Main content area - Split into columns
col1, col2 = st.columns([3, 2])

# Left column - Transcript
with col2:
    st.header("Live Transcript")
    
    # Status indicator
    if st.session_state.recording:
        st.info("Recording and transcribing in progress...")
    
    # Use a simple text area with fixed height instead of a dynamic placeholder
    # This is more stable for long, frequently updating content
    transcript_container = st.container(height=400)
    
    with transcript_container:
        # Render the transcript as HTML within markdown
        # This avoids the ElementNode issue with st.empty() updates
        if st.session_state.transcript:
            formatted = format_transcript_with_highlight(
                st.session_state.transcript,
                st.session_state.last_correction
            )
            st.markdown(f'<div style="height: 380px; overflow-y: auto;">{formatted}</div>', unsafe_allow_html=True)
        else:
            st.write("*Waiting for transcript...*")
    
    

# Right column - Summary in Markdown format
with col1:
    st.header("Meeting Summary")
    summary_container = st.container(height=400)
    
    with summary_container:
        if st.session_state.current_summary:
            # Display timestamp of last update
            last_update = getattr(st.session_state, 'last_summary_timestamp', 'N/A')
            st.caption(f"Last updated at {last_update}")
            
            # Check if we have a highlighted summary
            if hasattr(st.session_state, 'highlighted_summary') and st.session_state.highlighted_summary:
                # Display the highlighted summary
                st.markdown(
                    f'<div style="height: 380px; overflow-y: auto;">{st.session_state.highlighted_summary}</div>', 
                    unsafe_allow_html=True
                )
            else:
                # Display the regular summary if no highlights are available
                st.markdown(
                    f'<div style="height: 380px; overflow-y: auto;">{st.session_state.current_summary}</div>', 
                    unsafe_allow_html=True
                )
        else:
            st.write("*No summary generated yet. Start recording to generate a summary.*")
    
    # Question and answer section
    st.header("Ask a Question")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Enter your question about the meeting content")
        submit_question = st.form_submit_button("Submit Question")
        
    if submit_question and question:
        answer = answer_question(question)
        st.write("Answer:")
        st.info(answer)
        
    # Previous Q&A display
    if st.session_state.questions_answers:
        st.header("Previous Questions & Answers")
        qa_container = st.container(height=300)
        with qa_container:
            for qa in reversed(st.session_state.questions_answers):
                with st.expander(f"Q&A at {qa['timestamp']}", expanded=False):
                    st.write(f"**Question:** {qa['question']}")
                    st.write(f"**Answer:** {qa['answer']}")

# Footer with system information
st.markdown("---")
with st.expander("System Information"):
    if torch.cuda.is_available():
        st.write(f"CUDA available: Yes (Device: {torch.cuda.get_device_name(0)})")
    else:
        st.write("CUDA available: No (using CPU for processing)")
    
    st.write(f"Whisper model: {st.session_state.whisper_selection}")
    st.write(f"LLM provider: {st.session_state.llm_provider} ({st.session_state.llm_model})")
    st.write(f"Summary frequency: {st.session_state.summary_frequency} seconds")

# Main loop for handling results from threads and updating UI
if st.session_state.recording:
    rerun_needed = False
    # Process new transcript text from the queue
    while not st.session_state.transcript_q.empty():
        try:
            # Get new transcription result
            result = st.session_state.transcript_q.get_nowait()
            
            # Extract the transcribed text
            text = result["text"] if isinstance(result, dict) else result
            
            # Find what's truly new in this transcription
            new_content, is_correction = find_new_content(
                st.session_state.transcript, 
                text
            )
            
            # Only update if we have meaningful new content
            if new_content:
                if is_correction:
                    # Handle correction case - find where to replace
                    common_suffix = find_common_suffix(st.session_state.transcript, new_content, max_words=15)
                    if common_suffix:
                        correction_point = st.session_state.transcript.rfind(common_suffix)
                        if correction_point > 0:
                            st.session_state.transcript = st.session_state.transcript[:correction_point] + new_content
                            st.session_state.last_correction = new_content
                            rerun_needed = True
                    else:
                        # Fallback if no good correction point
                        st.session_state.transcript += " " + new_content
                        st.session_state.last_correction = None
                        rerun_needed = True
                else:
                    # Simple append for new content
                    st.session_state.transcript += " " + new_content
                    st.session_state.last_correction = None
                    rerun_needed = True
                
        except queue.Empty:
            break

    # Check if it's time to generate a summary
    current_time = time.time()
    if (current_time - st.session_state.last_summary_time >= st.session_state.summary_frequency and
        st.session_state.transcript.strip()):
        generate_summary()
        st.session_state.last_summary_time = current_time
        rerun_needed = True

    # Only rerun if needed, and not too frequently
    if rerun_needed:
        time.sleep(0.25)  # Short sleep to prevent excessive CPU usage
        st.rerun()
    else:
        time.sleep(0.5)  # Longer sleep if no updates
        st.rerun() 