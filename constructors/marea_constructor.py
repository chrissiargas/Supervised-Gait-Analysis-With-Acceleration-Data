import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import io

import matplotlib.pyplot as plt

# functions to create binary variable and index
def get_binary(subject_data, activity_as_string, start_time, end_time):
    temp_list = []
    for i in range(len(subject_data)):
        if start_time <= i + 1 <= end_time:
            temp_list.append(1)
        else:
            temp_list.append(0)
    subject_data[activity_as_string] = temp_list

    return subject_data


def get_index(subject_data, activity_as_string, start_time, end_time):
    counter = 1
    temp_list = []
    for i in range(len(subject_data)):
        if start_time <= i + 1 <= end_time:
            temp_list.append(counter)
            counter += 1
        else:
            temp_list.append(0)
    subject_data[activity_as_string + '_index'] = temp_list

    return subject_data


def get_hsto(subject_data, GT, foot_event_as_string, activity_as_string):
    event_as_string = foot_event_as_string + '_' + activity_as_string
    timestamps = pd.DataFrame(GT[foot_event_as_string][0, 0], columns=[event_as_string])
    timestamps['indicator'] = 1

    subject_data = pd.merge(subject_data, timestamps[[event_as_string, 'indicator']],
                            left_on=activity_as_string + '_index', right_on=event_as_string, how='left')

    subject_data = subject_data.drop(event_as_string, axis=1)
    subject_data = subject_data.fillna(0)

    subject_data[foot_event_as_string] += subject_data['indicator']
    subject_data = subject_data.drop('indicator', axis=1)

    return subject_data


class marea:
    def __init__(self):
        self.source_path = os.path.join(
            os.path.expanduser('~'),
            'datasets',
            'MAREA'
        )

        self.target_path = os.path.join(os.path.expanduser('~'),
                                      'datasets','MAREA_new')

        z = os.path.join(self.source_path, 'extracted')

        try:
            shutil.rmtree(z)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    def __call__(self, *args, **kwargs):

        subject_ids = [*range(1, 21, 1)]
        combined_data = None

        # Choose the subject ID whose data you want to convert
        for id, subject_id in enumerate(tqdm(subject_ids)):
            if subject_id == 4:
                continue

            subject_id = str(subject_id)

            subject_data = self.get_IMU(subject_id)
            subject_data = self.get_activity(subject_data, subject_id)
            subject_data = self.get_events(subject_data, subject_id)

            subject_data.to_csv(os.path.join(self.target_path, "Sub_" + subject_id + ".csv"), encoding='utf-8')

            subject_data['subject'] = subject_id
            if id == 0:
                combined_data = subject_data
            else:
                combined_data = pd.concat([combined_data, subject_data], axis=0, ignore_index=True)

        combined_data.to_csv(os.path.join(self.target_path, "marea_full" + ".csv"), encoding='utf-8')

        return combined_data

    def get_IMU(self, subject_id):

        # The readings from each accelerometer (LF, RF, Waist and Wrist) are stored
        # in separate text files for each subject
        # Firstly, we combine these data into a single table
        RECORDINGS_PATH = os.path.join(self.source_path, 'Subject Data_txt format')
        LF_FILE = os.path.join(RECORDINGS_PATH, 'Sub' + subject_id + '_LF.txt')
        RF_FILE = os.path.join(RECORDINGS_PATH, 'Sub' + subject_id + '_RF.txt')
        Waist_FILE = os.path.join(RECORDINGS_PATH, 'Sub' + subject_id + '_Waist.txt')
        Wrist_FILE = os.path.join(RECORDINGS_PATH, 'Sub' + subject_id + '_Wrist.txt')  # Comment out for subject 4

        # read in the data into dataframe
        LF_DATA = pd.read_csv(LF_FILE, header=0)
        RF_DATA = pd.read_csv(RF_FILE, header=0)
        Waist_DATA = pd.read_csv(Waist_FILE, header=0)
        Wrist_DATA = pd.read_csv(Wrist_FILE, header=0)  # Comment out for subject 4

        # Since the column headings are accX, accY, accZ, we will need to rename them to know which accelerometer they came from
        # To that we add a "_LF/RF/Waist/Wrist"
        LF_DATA = LF_DATA.rename(index=str, columns={"accX": "accX_LF", "accY": "accY_LF", "accZ": "accZ_LF"})
        RF_DATA = RF_DATA.rename(index=str, columns={"accX": "accX_RF", "accY": "accY_RF", "accZ": "accZ_RF"})
        Waist_DATA = Waist_DATA.rename(index=str,
                                         columns={"accX": "accX_Waist", "accY": "accY_Waist", "accZ": "accZ_Waist"})
        Wrist_DATA = Wrist_DATA.rename(index=str, columns={"accX": "accX_Wrist", "accY": "accY_Wrist",
                                                             "accZ": "accZ_Wrist"})  # Comment out for subject 4

        # Merge the columns together
        subject_data = pd.concat([LF_DATA, RF_DATA, Waist_DATA, Wrist_DATA], axis=1, sort=False)
        subject_data['time'] = subject_data.index

        return subject_data


    def get_activity(self, subject_data, subject_id):
        # The Activity Timings dataset shows when the subject is carrying out a particular activity (Walk/Run).
        # We will look up the timings for each subject and create a binary variable for each activity to indicate
        # whether that activity is currently being carried out
        # We will also add in the sample number for each activity.
        # These fields end with "index"
        activity_path = os.path.join(self.source_path, 'Activity Timings')

        if int(subject_id) < 12:
            # INDOORS

            # # read in the data into dataframe
            timings_data = pd.read_csv(activity_path + '/Indoor Experiment Timings.txt', sep=',', header=None)
            timings_data.columns = ["Walk", "Walk_End", "Walk_Run", "Slope", "Slope_End", "Flat", "Flat_End",
                                    "Flat_Run"]

            # # Get subject-specific timings
            walk = int(timings_data['Walk'][timings_data.index[int(subject_id) - 1]])
            walk_end = int(timings_data['Walk_End'][timings_data.index[int(subject_id) - 1]])
            walk_run = int(timings_data['Walk_Run'][timings_data.index[int(subject_id) - 1]])
            slope = int(timings_data['Slope'][timings_data.index[int(subject_id) - 1]])
            slope_end = int(timings_data['Slope_End'][timings_data.index[int(subject_id) - 1]])
            flat = int(timings_data['Flat'][timings_data.index[int(subject_id) - 1]])
            flat_end = int(timings_data['Flat_End'][timings_data.index[int(subject_id) - 1]])
            flat_run = int(timings_data['Flat_Run'][timings_data.index[int(subject_id) - 1]])

            # treadmill_walk
            subject_data = get_binary(subject_data, 'treadmill_walk', walk, walk_end)
            subject_data = get_index(subject_data, 'treadmill_walk', walk, walk_end)

            # treadmill_walknrun
            subject_data = get_binary(subject_data, 'treadmill_walknrun', walk, walk_run)
            subject_data = get_index(subject_data, 'treadmill_walknrun', walk, walk_run)

            # treadmill_slope_walk
            subject_data = get_binary(subject_data, 'treadmill_slope_walk', slope, slope_end)
            subject_data = get_index(subject_data, 'treadmill_slope_walk', slope, slope_end)

            # indoor_walk
            subject_data = get_binary(subject_data, 'indoor_walk', flat, flat_end)
            subject_data = get_index(subject_data, 'indoor_walk', flat, flat_end)

            # indoor_walknrun
            subject_data = get_binary(subject_data, 'indoor_walknrun', flat, flat_run)
            subject_data = get_index(subject_data, 'indoor_walknrun', flat, flat_run)

        else:
            # OUTDOORS

            # read in the data into dataframe
            timings_data = pd.read_csv(activity_path + '/Outdoor Experiment Timings.txt', sep=',', header=None)
            timings_data.columns = ["Walk", "Walk_End", "Walk_Run"]

            # Get subject-specific timings
            walk = int(timings_data['Walk'][timings_data.index[int(subject_id) - 12]])
            walk_end = int(timings_data['Walk_End'][timings_data.index[int(subject_id) - 12]])
            walk_run = int(timings_data['Walk_Run'][timings_data.index[int(subject_id) - 12]])

            # outdoor_walk
            subject_data = get_binary(subject_data, 'outdoor_walk', walk, walk_end)
            subject_data = get_index(subject_data, 'outdoor_walk', walk, walk_end)

            # outdoor_walknrun
            subject_data = get_binary(subject_data, 'outdoor_walknrun', walk, walk_run)
            subject_data = get_index(subject_data, 'outdoor_walknrun', walk, walk_run)

        return subject_data

    def get_events(self, subject_data, subject_id):
        # Read the data
        EVENTS_PATH = os.path.join(self.source_path, 'GroundTruth.mat')
        mat = io.loadmat(EVENTS_PATH)
        events = mat['GroundTruth']

        foot_events = ['LF_HS', 'RF_HS', 'LF_TO', 'RF_TO']
        for foot_event in foot_events:
            subject_data[foot_event] = 0

        if int(subject_id) < 12:
            # Indoors
            TREADMILL_WALKNRUN_GT = events['treadWalknRun'][0, int(subject_id) - 1]
            TREADMILL_SLOPE_WALK_GT = events['treadIncline'][0, int(subject_id) - 1]
            INDOOR_WALKNRUN_GT = events['indoorWalknRun'][0, int(subject_id) - 1]

            GTs = [TREADMILL_WALKNRUN_GT, TREADMILL_SLOPE_WALK_GT, INDOOR_WALKNRUN_GT]
            activities = ['treadmill_walknrun', 'treadmill_slope_walk', 'indoor_walknrun']

            for GT, activity in zip(GTs, activities):
                for foot_event in foot_events:
                    subject_data = get_hsto(subject_data, GT, foot_event, activity)

        else:
            # Outdoors
            OUTDOOR_WALKNRUN_GT = events['outdoorWalknRun'][0, int(subject_id) - 12]
            activity = 'outdoor_walknrun'

            for foot_event in foot_events:
                subject_data = get_hsto(subject_data, OUTDOOR_WALKNRUN_GT, foot_event, activity)

        return subject_data

if __name__ == '__main__':
    constructor = marea()
    constructor()
