<div align="center">
  <br />
  <p>
    <a href="https://github.com/Chokyotager/LectureVaporiserTool-2"><img src="/art/banner.png" alt="art" /></a>
  </p>
  <br />
  <p>
  </p>
</div>
(Art generated using StableDiffusion)

## About
**Lecture Vaporiser Tool 2 (LVT-2)** is a system integrating multiple AI systems to effectively compress videos of lectures into notes. It can hear what is being said in the lecture, and read slides if presented.

The idea was to consolidate what was being said and taught, without adding extra content into notes, much like how a student attending the lecture would use this.

## Example notes
You can see a generated example output in the demo folder. This was run on MIT's OpenCourseWare "1. Introduction to Computational and Systems Biology" (https://www.youtube.com/watch?v=lJzybEXmIj0) by Prof. Christopher Burge licensed under BY-NC-SA.

## Limitations and liability
LVT-2 occassionally can get the names of people and jargon wrong. If the slides presented spell these text right, it may be more likely to correct this in context. If the lecture has many repeated points, the notes will likewise chronologically show these repeats (much like a student attending the lecture in person!). Depending on the verbosity of the lecture, adjusting hyperparameters will be necessary. Some concepts may also be incomplete or incorrect, though this is generally rare. Please exercise your own discretion and due dilligence when using the generated notes, I am NOT liable for the stuff this program spews!

## Installation
There are a few tools which you need installed on your system:

1. FFMPEG
2. OpenAI's Whisper (may have other requirements on your system!)
3. EasyOCR (may have other requirements on your system!)

Python dependencies:

```sh
git clone https://github.com/Chokyotager/LectureVaporiserTool-2
pip install -r requirements.txt
```

These should be accessible as executables (i.e. "ffmpeg", "whisper") when keyed into the command line. The code directly references these
executables and you will have to change them if they are located elsewhere.

## Running

`python3 lvt.py --key <OPENAI KEY> input.mp4 output`

You do not have to specify the `OPENAI_KEY` if you have an environmental variable `OPENAI_API_KEY` defined.

The output summary would be in the folder. It takes varying amounts of time to generate these notes, namely depdendent on how "crowded" the slides are. Slides that are full of text will take significantly longer for the OCR to read.

## Contribution

You are welcome to contribute through pull requests or forking.

## License
License details can be found in the LICENSE file.
