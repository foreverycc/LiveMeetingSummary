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

# LLM provider configuration
@st.cache_resource
def load_api_keys():
    """Load API keys from environment variables or Streamlit secrets."""
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
    }
    
    # Initialize API clients
    if keys["GOOGLE_API_KEY"]:
        genai.configure(api_key=keys["GOOGLE_API_KEY"])
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
    """Generate a summary of the current transcript using the selected LLM."""
    if not st.session_state.transcript.strip():
        return
    
    try:
        summary = ""
        prompt = f"""Please provide a concise summary of the following meeting transcript:
        
{st.session_state.transcript}

Summary:"""

        # Use selected LLM to generate summary
        if st.session_state.llm_provider == "Gemini":
            model = genai.GenerativeModel(st.session_state.llm_model)
            response = model.generate_content(prompt)
            summary = response.text
        elif st.session_state.llm_provider == "OpenAI":
            response = openai.ChatCompletion.create(
                model=st.session_state.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            summary = response.choices[0].message.content
        
        # Store the summary with timestamp
        if summary:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.summaries.append({
                "timestamp": timestamp,
                "summary": summary
            })
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
        "summaries": st.session_state.summaries,
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
        st.session_state.summaries = []
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
    if last_updated_portion and last_updated_portion in transcript:
        # Highlight the corrected portion
        highlighted = transcript.replace(
            last_updated_portion, 
            f'<span class="correction">{last_updated_portion}</span>', 
            1
        )
        return highlighted
    return transcript

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
with col1:
    st.header("Live Transcript")
    
    # Status indicator
    if st.session_state.recording:
        status_placeholder = st.empty()
        status_placeholder.info("Recording and transcribing in progress...")
    
    # Create a placeholder for the transcript
    if 'transcript_placeholder' not in st.session_state:
        st.session_state.transcript_placeholder = st.empty()
    
    # Only initializing or updating the placeholder, not creating new ones on rerun
    # This provides a more stable visual experience
    st.session_state.transcript_placeholder.markdown(st.session_state.transcript)
    
    # Question and answer section
    st.header("Ask a Question")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Enter your question about the meeting content")
        submit_question = st.form_submit_button("Submit Question")
        
    if submit_question and question:
        answer = answer_question(question)
        st.write("Answer:")
        st.info(answer)

# Right column - Summaries
with col2:
    st.header("Meeting Summaries")
    summaries_container = st.container(height=400)
    with summaries_container:
        for summary in reversed(st.session_state.summaries):
            with st.expander(f"Summary at {summary['timestamp']}", expanded=True):
                st.write(summary["summary"])
    
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
            
            if isinstance(result, dict) and "is_correction" in result and result["is_correction"]:
                # This is a correction to previous text
                new_text = result["text"]
                current_text = st.session_state.transcript
                
                # A simple approach: find the best match for the last N words
                words = current_text.split()
                if len(words) > 10:
                    # Find the best matching point to apply correction
                    common_suffix = find_common_suffix(current_text, new_text, max_words=15)
                    if common_suffix:
                        # Replace the last portion of the transcript with the corrected version
                        correction_point = current_text.rfind(common_suffix)
                        if correction_point > 0:
                            st.session_state.transcript = current_text[:correction_point] + new_text
                            rerun_needed = True
                    else:
                        # Fall back to appending if we can't find a good correction point
                        st.session_state.transcript += " " + new_text
                        rerun_needed = True
                else:
                    # Not enough context for correction, just append
                    st.session_state.transcript += " " + new_text
                    rerun_needed = True
            else:
                # Simple append for non-correction segments
                text = result["text"] if isinstance(result, dict) else result
                st.session_state.transcript += " " + text
                rerun_needed = True
                
            # Format with highlighting if this was a correction
            formatted_transcript = format_transcript_with_highlight(
                st.session_state.transcript,
                last_updated_portion=new_text if isinstance(result, dict) and "is_correction" in result and result["is_correction"] else None
            )
            st.session_state.transcript_placeholder.markdown(
                formatted_transcript, 
                unsafe_allow_html=True
            )
                
        except queue.Empty:
            break

    # Check if it's time to generate a summary
    current_time = time.time()
    if (current_time - st.session_state.last_summary_time >= st.session_state.summary_frequency and
        st.session_state.transcript.strip()):
        generate_summary() # Call from main thread is safe
        st.session_state.last_summary_time = current_time
        rerun_needed = True

    # Ensure we rerun the app periodically to keep the UI responsive even when no updates occur
    time.sleep(0.25)  # Short sleep to prevent excessive CPU usage
    st.rerun() 