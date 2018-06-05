"""

Filters LOS scenarios and scenarios when too many resolutions were necessary.
Removes resolution txt files and SSD png files

"""

from glob import glob
from os import remove


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def filter_data(folder='data'):
    # make list of scenarios with too many resolutions
    filelist = glob('{}/resolutions*.txt'.format(folder))
    remove_list = []
    for fname in filelist:

        # save first resolution in file
        with open(fname, 'r') as file:
            content = file.readlines()
            hdg_resolution = float(content[1].partition(";")[0])  # only take first resolution\
            #print(hdg_resolution)

        # remove if more than 3 resolutions or if first resolution is 0 deg
        file_length = file_len(fname)
        if file_length > 4 or hdg_resolution == 0:
            scenario_num = fname.partition('_S')[2]
            scenario_num = scenario_num[:-4]
            remove_list.append(scenario_num)

    # make list of scenarios that had LOS
    LOS_list = []
    try:
        with open('{}/LOS_tracker.txt'.format(folder),'r') as file:
            content = file.readlines()
            for line in content:
                scenario_num = line[6:-1]
                LOS_list.append(scenario_num)

        # combine lists
        for scenario_num in LOS_list:
            if scenario_num not in remove_list:
                remove_list.append(scenario_num)
    except FileNotFoundError:
        pass

    # remove reso txt files
    for scenario in remove_list:
        fname = '{}/resolutions_S{}.txt'.format(folder, scenario)
        try:
            remove(fname)
            print('Removed: {}'.format(fname))
        except FileNotFoundError:
            pass

    # remove SSDs
    for scenario in remove_list:
        if scenario is not None:
            sublist = glob('{}/SSD_S{}_T*.png'.format(folder, scenario))
            for subitem in sublist:
                remove(subitem)
                print('Removed: {}'.format(subitem))

    resos_removed = len(remove_list)
    print('Scenarios removed: {}'.format(resos_removed))


if __name__ == "__main__":
    folder = 'test_data_filtered'
    filter_data(folder)
