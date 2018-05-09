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
        file_length = file_len(fname)
        if file_length > 4:
            scenario_num = fname.partition('_S')[2]
            scenario_num = scenario_num[:-4]
            remove_list.append(scenario_num)

    # make list of scenarios that had LOS
    LOS_list = []
    with open('{}/LOS_tracker.txt'.format(folder),'r') as file:
        content = file.readlines()
        for line in content:
            scenario_num = line[6:-1]
            LOS_list.append(scenario_num)

    # combine lists
    for scenario_num in LOS_list:
        if scenario_num not in remove_list:
            remove_list.append(scenario_num)

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


if __name__ == "__main__":
    folder = 'data'
    filter_data(folder)
