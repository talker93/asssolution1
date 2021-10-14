import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


DATASET_PATH = "workspace"
BLOCK_SIZE = 1024
HOP_SIZE = 512


def ceil(number):
    return int(-1 * number // 1 * -1)


def generate_sin(frequency, sample_rate, length):
    return np.cos(2 * np.pi * frequency * np.arange(length) / sample_rate)


def convert_freq2midi(freqInHz):
    midi = 69 + (np.log2(np.asanyarray(freqInHz)) - np.log2(440)) * 12
    return midi


def block_audio(x, blockSize, hopSize, fs):
    block_numbers = ceil(x.size / hopSize)
    print("x.size", x.size)
    print("block_numbers", block_numbers)
    xb = np.zeros((block_numbers, blockSize))
    timeInSec = np.zeros(block_numbers)
    row = 0
    while row < block_numbers:
        if blockSize <= x[row*hopSize:].size:
            xb[row] = x[(row * hopSize):(row * hopSize + blockSize)]
            timeInSec[row] = row * hopSize / fs
        else:
            xb[row] = np.append(x[(row*hopSize):],np.zeros(blockSize-(x.size-(row*hopSize))))
        row = row + 1
    print("xb.shape", xb.shape)
    print("timeInSec", timeInSec.shape)
    return xb, timeInSec


def comp_acf(inputVector, blsNormalized=True):
    inputVector_corr = np.zeros(inputVector.size)
    j = 0
    while j < inputVector.size:
        product_sum = 0
        # the old way to get convolution by two loop, really slow
        # while i < inputVector.size - j:
        #     product = inputVector[i] * inputVector[i + j]
        #     product_sum = product + product_sum
        #     i = i + 1
        product_sum = np.dot(inputVector[:len(inputVector)-j], inputVector[j:])
        inputVector_corr[j] = product_sum
        j = j + 1
    if blsNormalized == 1:
        inputVector_corr = inputVector_corr / inputVector_corr[0]
    return inputVector_corr


def get_f0_from_acf(r, fs):
    high_time = fs/50
    low_time = fs/2000
    r[int(high_time):] = 0
    r[:int(low_time)] = 0
    r_argmax = r.argmax()
    r_estimate = float(fs) / r_argmax
    return r_estimate


def track_pitch_acf(x, blockSize, hopSize, fs):
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    blocks_corr = np.zeros((xb.shape[0], xb.shape[1]))
    blocks_f0 = np.zeros(xb.shape[0])
    i = 0
    while i < xb.shape[0]:
        blocks_corr[i] = comp_acf(xb[i], True)
        blocks_f0[i] = get_f0_from_acf(blocks_corr[i], fs)
        timeInSec[i] = (i * hopSize) / fs
        print("Block at", timeInSec[i] ,"s, F0 is:", blocks_f0[i], " Hz")
        i = i + 1
    return blocks_f0, timeInSec


def read_pitch_txt(fileAddr):
    ground_truth_F0 = []
    with open(fileAddr, "r") as file:
        rows = file.readlines()
        for row in rows:
            row_content = row.split("     ")
            ground_truth_F0.append(float(row_content[2]))
    return ground_truth_F0


def eval_pitchtrack(estimateInHz, groundtruthInHz):
    errorCent = np.zeros(estimateInHz.size)
    estimateInPitch = np.zeros(estimateInHz.size)
    groundtruthInPitch = np.zeros(groundtruthInHz.size)
    errorCent_normal = np.zeros(errorCent.size)
    i = 0
    while i < estimateInHz.shape[0] - 1:
        estimateInPitch[i] = convert_freq2midi(estimateInHz[i])
        groundtruthInPitch[i] = convert_freq2midi(groundtruthInHz[i])
        errorCent[i] = 100 * (estimateInPitch[i] - groundtruthInPitch[i])
        i = i + 1
    errCentRms = np.sqrt(np.mean(errorCent**2))
    print("errCentRms:", errCentRms)
    return errCentRms


def data_process(blocks_F0, ground_truth_F0, blocks_time):
    blocks_F0_list = blocks_F0.tolist()
    blocks_time_list = blocks_time.tolist()
    # transform to list to drop zero data
    j = 0
    while j < len(ground_truth_F0):
        if ground_truth_F0[j] <= 0:
            del ground_truth_F0[j]
            del blocks_F0_list[j]
            blocks_time_list[j] = 0
            j = j - 1
        j = j + 1
    n = 0
    while n < len(blocks_F0_list) - 1:
        if blocks_F0_list[n] <= 0:
            del ground_truth_F0[n]
            del blocks_F0_list[n]
            blocks_time_list[j] = 0
            n = n - 1
        n = n + 1
    # drop start and end blocks
    # k = 1
    # while k < 30:
    #     del ground_truth_F0[0]
    #     del ground_truth_F0[len(ground_truth_F0)-1]
    #     del blocks_F0_list[0]
    #     del blocks_F0_list[len(blocks_F0_list)-1]
    #     k = k + 1
    blocks_F0_processed = np.array(blocks_F0_list)
    ground_truth_F0_processed = np.array(ground_truth_F0)
    blocks_time_processed = np.array(blocks_time_list)
    return blocks_F0_processed, ground_truth_F0_processed, blocks_time_processed


def run_evaluation(complete_path_to_data_folder):
    print("                                                                ")
    print("                                                                ")
    print("------------------------Evaluation start!-----------------------")
    print("                                                                ")
    print("                                                                ")
    mapping = []
    for root, dirs, filenames in os.walk(complete_path_to_data_folder):
        for file in filenames:
            mapping.append(os.path.join(root, file))
    t = 1
    for txt_path in mapping:
        if os.path.splitext(txt_path)[1] == ".txt":
            print("------------------------Evaluating file", t, "---------------------------")
            groundTruthF0 = read_pitch_txt(txt_path)
            print(txt_path, ": success read!")
            tup = os.path.splitext(txt_path)[0]
            str = ''
            for item in tup:
                str = str + item
                str = str.replace('.f0.Corrected', '')
            wav_path = str + ".wav"
            sampleRate, audio = wavfile.read(wav_path)
            print(wav_path, ": success read!")
            blocksF0, blocksTime = track_pitch_acf(audio, BLOCK_SIZE, HOP_SIZE, sampleRate)
            blocksF0Processed, groundTruthF0Processed, blocksTimeProcessed = data_process(blocksF0, groundTruthF0, blocksTime)
            errCentRms = eval_pitchtrack(blocksF0Processed, groundTruthF0Processed)
            t = t + 1
    print("                                                                   ")
    print("                                                                   ")
    print("------------------------Evaluation finished!-----------------------")
    print("                                                                   ")
    print("                                                                   ")


def sineF0_evaluation():
    print("                                                                ")
    print("                                                                ")
    print("------------------------Sine Evaluation start!------------------")
    print("                                                                ")
    print("                                                                ")
    sineSignal_441 = generate_sin(441, 44100, 88200)
    sineSignal_882 = generate_sin(882, 44100, 88200)
    sineSignal_441[44100:88200] = 0
    sineSignal_882[0:44100] = 0
    sineSignal = sineSignal_441 + sineSignal_882
    sineF0, sineTime = track_pitch_acf(sineSignal, BLOCK_SIZE, HOP_SIZE, 44100)
    plt.figure(figsize=(14, 5))
    plt.plot(sineF0[:sineF0.size])
    plt.title('Sine Signal F0 Estimation')
    plt.xlabel('Block Address')
    plt.ylabel('Frequency / Hz')
    errorInHerz = sineF0
    i = 0
    while i < sineF0.size:
        if i < ceil(sineF0.size / 2):
            errorInHerz[i] = sineF0[i] - 441.0
        else:
            errorInHerz[i] = sineF0[i] - 882.0
        i = i + 1
    # plt.figure(figsize=(14, 5))
    # plt.plot(errorInHerz[:sineF0.size])
    # plt.title('Error of Sine F0 Estimation')
    # plt.xlabel('Block Address')
    # plt.ylabel('Error (F0 - GroundTruth) / Hz')
    print("                                                                   ")
    print("                                                                   ")
    print("------------------------Sine Evaluation finished!------------------")
    print("                                                                   ")
    print("                                                                   ")


if __name__ == '__main__':
    # sineF0_evaluation()
    run_evaluation(DATASET_PATH)
    # plt.show()

# File 1. 01-D_AMairena.wav
# errCentRms = 220.68

# File 2. 24-M1_AMairena-Martinete.wav
# errCentRms = 524.45

# File 3. 63-M2_AMairena.wav
# errCentRms = 338.92
