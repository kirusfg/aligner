from split_audio_from_vtt import make_corpus
import os,sys
import subprocess
from corpus_to_tuples import get_tuples
import h5py
import numpy as np
import argparse

MFA_ALIGNER = '/raid/kirill_kirillov/.conda/cuda/bin/'
G2P_MODEL = 'english_us_mfa'
LANG_MODEL = 'english_mfa_lm'

def output_h5py(intervals, work_dir, video_id, name):
    data = {}
    data[video_id] = {}
    data[video_id]['intervals'] = np.array([[a[0][0], a[0][1]] for a in intervals])
    data[video_id]['features'] = np.array([[a[1].encode('utf-8')] for a in intervals],dtype="a32")

    write5Handle=h5py.File(os.path.join(work_dir, video_id + "_" + name + ".h5py"),'w')

    vidHandle=write5Handle.create_group(video_id)
    vidHandle.create_dataset("features",data=data[video_id]["features"])
    vidHandle.create_dataset("intervals",data=data[video_id]["intervals"])
    write5Handle.close()

def align():
    # try:
    #     os.mkdir(video_id)
    # except Exception as e:
    #     print(video_id + " folder already found. Quitting.")
    #     sys.exit(0)

    # work_dir = video_id
    # # Download the video, wav, and transcription
    # dwnld(video_id, work_dir)

    work_dir = 'data'
    processed_dir = 'processed'

    transcription = os.path.join(work_dir, 'transcript.vtt')
    wav = os.path.join(work_dir, 'audio.wav')
    unaligned = os.path.join(work_dir, processed_dir, 'transcript_unaligned')

    # Split and assemble data into corpus accepted by MFA
    make_corpus(wav,transcription,unaligned)
    corpus = unaligned
    video_dict = os.path.join(work_dir, processed_dir, 'dict.txt')

    # Generate dictionary
    output_dir = os.path.join(work_dir, processed_dir, 'transcript_aligned')
    subprocess.check_call(MFA_ALIGNER + 'mfa align %s %s %s %s' % (
        corpus,
        'english_us_mfa',
        'english_mfa',
        output_dir,
    ), shell=True)
    # subprocess.check_call(MFA_ALIGNER + 'mfa_generate_dictionary ' + G2P_MODEL + ' ' + corpus + ' ' + video_dict,shell=True)

    # Run the alignment tool
    
    # align_call = MFA_ALIGNER + 'mfa_align ' + corpus + ' ' + video_dict + ' ' + LANG_MODEL + ' ' + output_dir
    # subprocess.check_call(align_call,shell=True)

    word_intervals,phone_intervals = get_tuples(output_dir)

    # Assemble the h5py files
    output_h5py(word_intervals, work_dir, processed_dir, "words")
    output_h5py(phone_intervals, work_dir, processed_dir, "phones")

    #print(word_intervals)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('video_id', type=str, help='The youtube ID of the video to be aligned.')
    # parser.add_argument('lang', type=str, help='The language code of the video.')
    # args = parser.parse_args()
    # align(args.video_id, args.lang)
    align()
