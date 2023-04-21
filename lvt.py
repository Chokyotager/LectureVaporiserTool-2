#/bin/python3

import os
import shutil
import json
import csv

import time
import math
import jaro

import argparse

parser = argparse.ArgumentParser(description="Run Lecture Vaporiser Tool 2 (LVT-2) by Hilbert Lam.")

if os.environ.get("OPENAI_API_KEY") is None:
    parser.add_argument("--key", action="store", required=True, type=str, help="OpenAI key")

else:
    parser.add_argument("--key", action="store", required=False, type=str, help="OpenAI key", default=os.environ["OPENAI_API_KEY"])

parser.add_argument("--gpt_temperature", action="store", type=float, help="ChatGPT temperature", default=0)
parser.add_argument("--gpt_presence_penalty", action="store", type=float, help="ChatGPT presence penalty", default=2)
parser.add_argument("--gpt_frequency_penalty", action="store", type=float, help="ChatGPT frequency penality", default=1)
parser.add_argument("--ocr_frame_time", action="store", type=float, help="Time (in seconds) between frames for OCR", default=10)
parser.add_argument("--sentences_per_chunk", action="store", type=int, help="Number of coherent sentences to pass per chunk into ChatGPT. Smaller values will create far more verbose notes.", default=100)

parser.add_argument("input", action="store", type=str, help="Input video file (can be mp4, ts, etc.)")
parser.add_argument("output", action="store", type=str, help="Output folder")

args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer

import language_tool_python

import torch

torch.backends.cudnn.enabled = False

import easyocr
import openai

lang_tool = language_tool_python.LanguageTool("en-US")

openai.api_key = args.key

output_directory = args.output + "/"
video = args.input

if not os.path.isfile(video):
    raise Exception("Video file does not exist!")

seconds_per_frame_transcription = args.ocr_frame_time

sentences_per_chunk = args.sentences_per_chunk
preliminary_summary_ratio = 0.55
preliminary_visual_ratio = 0.1
context_sentences = 15
sentences_for_question_generation = 50

gpt_engine = "gpt-3.5-turbo"
gpt_temperature = args.gpt_temperature
gpt_presence_penalty = args.gpt_presence_penalty
gpt_frequency_penalty = args.gpt_frequency_penalty

time_between_calls = 15

def transcribe_video (directory, video, seconds_per_frame=5):

    similarity_threshold = 0.8
    length_threshold = 5

    fps = 1 / seconds_per_frame

    if os.path.isdir(directory + "images"):
        shutil.rmtree(directory + "images")

    os.makedirs(directory + "images", exist_ok=True)

    os.system(f"ffmpeg -i {video} -vf fps={fps} -dpi 800 {directory}/images/out_%d.png")

    # FFMPEG to split images
    frames = sorted(os.listdir(directory + "images/"), key=lambda x: int(x.replace(".png", "").split("_")[1]))

    transcripts = list()
    last_transcript = str()

    def correct_text (text):

        matches = lang_tool.check(text)

        corrections = list()
        mistakes = list()
        errorStartPosition = list()
        errorEndPosition = list()

        for match in matches:
            # To check if there are any correct replacement available
            if len(match.replacements) > 0:
                errorStartPosition.append(match.offset)
                errorEndPosition.append(match.offset + match.errorLength)
                mistakes.append(text[match.offset : match.offset + match.errorLength])
                corrections.append(match.replacements[0])

        # Converting our originaltext into list
        newText = list(text)

        for i in range(len(errorStartPosition)):
            for j in range(len(text)):
                newText[errorStartPosition[i]] = corrections[i]
                if j > errorStartPosition[i] and j < errorEndPosition[i]:
                    newText[j] = ""

        # Joining our list to convert to string
        newText = "".join(newText)
        return newText

    reader = easyocr.Reader(["en"], gpu=True)

    frame_number = 0

    for frame in tqdm(frames):

        timeframe = frame_number * seconds_per_frame

        txt = reader.readtext(directory + "images/" + frame, detail=0)

        current_transcript = (" ".join(txt)).strip()
        current_transcript = current_transcript.replace("\n", " ").strip()

        current_transcript = correct_text(current_transcript)

        if len(current_transcript) < length_threshold:
            continue

        if jaro.jaro_winkler_metric(current_transcript, last_transcript) >= similarity_threshold:
            continue

        transcripts.append({"timeframe": timeframe, "transcript": current_transcript})
        last_transcript = current_transcript

        frame_number += 1

    shutil.rmtree(directory + "images")

    output = json.dumps(transcripts, indent=4)
    open(directory + "video_ocr.json", "w+").write(output)

    return output

system_message = {"role": "system", "content": "You are a helpful AI assistant that summarises talks into concise, detailed and digestible point form explanatory notes given a template. You also pick up key terms that are in the talk and do not add in your own examples, neither do you miss out any key examples given."}

os.makedirs(output_directory, exist_ok=True)

os.system(f"ffmpeg -i {video} -vn {output_directory}/audio.wav")
os.system(f"whisper --language en --model base.en --threads 32 {output_directory}/audio.wav --output_dir {output_directory}")

transcribe_video(output_directory, video, seconds_per_frame=seconds_per_frame_transcription)

video_json = json.loads(open(output_directory + "video_ocr.json").read())
transcript = list(csv.reader(open(output_directory + "audio.tsv"), delimiter="\t"))[1:]

# Run preliminary summarisation
number_sentences = int()
current_transcript = list()

current_time_start = 0
current_time_end = int()

all_summaries = list()

for segment in transcript:

    time_start = float(segment[0]) / 1000
    time_end = float(segment[1]) / 1000

    text = segment[2]

    if len(text.split(" ")) < 3:
        continue

    sentences = text.split(". ")
    number_sentences += len(sentences)

    current_transcript.append(text)

    if number_sentences > sentences_per_chunk:

        current_time_end = time_end
        # Offset
        current_time_start -= seconds_per_frame_transcription

        video_transcript = ". ".join([x["transcript"] for x in video_json if (float(x["timeframe"]) >= current_time_start and float(x["timeframe"]) <= current_time_end)])

        sentence_amount = math.floor(preliminary_visual_ratio * len(video_transcript.split(". ")))
        parser = PlaintextParser.from_string(video_transcript, Tokenizer("english"))
        stemmer = Stemmer("english")
        summariser = LexRankSummarizer(stemmer)

        summarised_video_transcript = "(visual text: " + " ".join([str(x) for x in summariser(parser.document, sentence_amount)]) + ")"

        current_transcript = "\n ".join(current_transcript)
        sentence_amount = math.floor(preliminary_summary_ratio * number_sentences)

        # Start Sumy summarisation
        parser = PlaintextParser.from_string(current_transcript, Tokenizer("english"))
        stemmer = Stemmer("english")
        summariser = LexRankSummarizer(stemmer)

        summarised_sentences = " ".join([str(x) for x in summariser(parser.document, sentence_amount)])
        all_summaries.append(summarised_video_transcript + "\n" + summarised_sentences)

        current_transcript = list()
        number_sentences = int()

        current_time_start = time_start

if len(current_transcript) > 0:

    current_transcript = "\n ".join(current_transcript)
    sentence_amount = math.floor(preliminary_summary_ratio * number_sentences)

    # Start Sumy summarisation
    parser = PlaintextParser.from_string(current_transcript, Tokenizer("english"))
    stemmer = Stemmer("english")
    summariser = LexRankSummarizer(stemmer)

    summarised_sentences = " ".join([str(x) for x in summariser(parser.document, sentence_amount)])
    all_summaries.append(summarised_sentences)

# Get transcript context
parser = PlaintextParser.from_string("\n".join([x[2] for x in transcript if len(x[2].split(" ")) > 2]), Tokenizer("english"))
stemmer = Stemmer("english")
summariser = LexRankSummarizer(stemmer)

summarised_sentences = " ".join([str(x) for x in summariser(parser.document, context_sentences)])

context_prompt = """
I need you to summarise the following talk into 1-3 sentences, totalling not more than 50 words. This sentence should highlight the overall context of the talk. You should not output anything else but the summary itself.

{|INPUT|}
"""

context_prompt = context_prompt.replace("{|INPUT|}", summarised_sentences)

completion = openai.ChatCompletion.create(
    model=gpt_engine,
    messages=[{"role": "system", "content": "You are an AI tasked to understand and summarise a talk into its context."}, {"role": "user", "content": context_prompt}],
    max_tokens=512,
    presence_penalty=gpt_presence_penalty,
    frequency_penalty=gpt_frequency_penalty,
    n=1,
    temperature=gpt_temperature,
    user="VaporiserSystem"
)

context = completion.choices[0].message["content"]
print("==== CONTEXT OF TALK ====")
print(context)

time.sleep(time_between_calls)

# Parse into ChatGPT for summary

base_prompt = """
I am going to input a partial transcript of a talk that is partially summarised. Please generate for me some notes in British English with the rules: multiple subtitles can be added per output, and each subtitle should have the minimum of one note. The subtitle should be enclosed in square brackets, as shown in the template above, and should accurately summaarise the note points described in it. The primary note points should be kept short, with elaboration bullet points used to supplement them. These elaboration bullet points are necessary only when it helps to break down the note point better. A template of how the notes could look like:

[Subtitles enclosed in square brackets]
1. Note 1
a. Elaboration on note 1
b. Further elaboration on note 1

2. Note 2
a. Elaboration on note 2

[Subtitles enclosed in square brackets]
1. Note 3
a. Elaboration on note 3
b. Further elaboration on note 3

2. Note 4
a. Elaboration on note 4

[Subtitles enclosed in square brackets]
1. Note 5
a. Elaboration on note 5
b. Further elaboration on note 5

Note that there could be some errors in the spelling, or wrong words used (i.e. cat herrings instead of cadherin, Haji Fork instead of Hartree-Fock, goji apparatus instead of Golgi apparatus). Please correct them in the final notes so that they are correct and coherent. The talk could be very technical and transcription errors may have occured.

For instances in which you have no notes to add for that particular section, please output nothing, or whitespace, though this should only be in the case where there is really no notes to be added. Do not output conversational text. I will paste excerpts of the lecture summary transcript respectively. I will also paste your previous note output if any - from this you can choose to start a new subtitle, or you can choose to continue from the previous output without adding a new subtitle, and following the respective numbering. You should not output anything else but the notes, with absolutely nothing else. You should not repeat what was mentioned in the previous output in the current output, but you can add on to it as new points if there is new content. For example, if the previous note ended on point #2, you can add a new point #3 under the same subtitle.

It is very important that all the coherent examples and concepts/theories that are mentioned in the transcript are listed as part of the notes unless they are wrong. It is also important that you do not add additional examples that are not mentioned in the transcript.

Talk context:
{|CONTEXT|}

Previous note output:
{|PREVIOUS|}

The partial summarised transcript is as follows:
{|OUTPUT|}

Again, if you have nothing to add, please output nothing, or whitespace.
"""

previous_summary = "No previous output"
responses = list()

for summary in tqdm(all_summaries):

    prompt = base_prompt.replace("{|OUTPUT|}", summary).replace("{|PREVIOUS|}", previous_summary).replace("{|CONTEXT|}", context)

    completion = openai.ChatCompletion.create(
        model=gpt_engine,
        messages=[system_message, {"role": "user", "content": prompt}],
        max_tokens=768,
        presence_penalty=gpt_presence_penalty,
        frequency_penalty=gpt_frequency_penalty,
        n=4,
        stop=None,
        temperature=gpt_temperature,
        user="VaporiserSystem"
    )

    time.sleep(time_between_calls)

    response = completion.choices[0].message["content"]

    if response.strip().lower() == "no comment" or response.strip().lower() == "no comment.":
        continue

    previous_summary = response
    responses.append(response)

    print(response)

summary_raw = "\n".join(responses)
open(output_directory + "summary_raw.txt", "w+").write(summary_raw)

parser = PlaintextParser.from_string(summary_raw, Tokenizer("english"))
stemmer = Stemmer("english")
summariser = LsaSummarizer(stemmer)

summarised_notes = " ".join([str(x) for x in summariser(parser.document, sentences_for_question_generation)])

prompt = """
Create 10 MCQs, and 3 short questions, on the following summarised notes. MCQs should have the options in capitalised A, B, C, and D. Please also output the suggested answers at the bottom, under a section titled "Suggested answers". You do not need to provide answers. You also do not need to output anything else but the questions and answers themselves:

{|INPUT|}
"""
prompt = prompt.replace("{|INPUT|}", summarised_notes)

completion = openai.ChatCompletion.create(
    model=gpt_engine,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1024,
    presence_penalty=gpt_presence_penalty,
    frequency_penalty=gpt_frequency_penalty,
    n=4,
    stop=None,
    temperature=1,
    user="VaporiserSystem"
)

response = completion.choices[0].message["content"]
open(output_directory + "generated_questions.txt", "w+").write(response)
