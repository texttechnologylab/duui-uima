import logging

logger = logging.getLogger(__name__)

def read_kaldi_format(filename, return_as_dict=True, values_as_string=False):
    key_list = []
    value_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            splitted_line = line.split()
            if len(splitted_line) == 2:
                key_list.append(splitted_line[0].strip())
                value_list.append(splitted_line[1].strip())
            elif len(splitted_line) > 2:
                key_list.append(splitted_line[0].strip())
                if values_as_string:
                    value_list.append(' '.join([x.strip() for x in splitted_line[1:]]))
                else:
                    value_list.append([x.strip() for x in splitted_line[1:]])
    if not return_as_dict:
        return key_list, value_list
    return dict(zip(key_list, value_list))


def save_kaldi_format(data, filename):
    if isinstance(data, list):
        if len(data) == 2:
            data = dict(zip(data[0], data[1]))
        elif isinstance(data[0], tuple) and len(data[0]) == 2:
            data = dict(data)
    with open(filename, 'w', encoding='utf-8') as f:
        for key, value in sorted(data.items(), key=lambda x: x[0]):
            if isinstance(value, list):
                value = ' '.join(value)
            try:
                #value = value.encode('utf-8')
                f.write(f'{key} {value}\n')
            except UnicodeEncodeError:
                logger.error(f'{key} {value}')
                raise

