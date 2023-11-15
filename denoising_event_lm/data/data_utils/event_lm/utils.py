
def print_stat_with_posneg(data, print_func=print):
    print_func("Num Seq: {:d}".format(len(data)))
    print_func("Avg Seq Length: {:5.3f}".format(sum(len(d['varg_seq']) for d in data) / len(data)))
    print_func("Num POS Seq: {:d}".format(len([d for d in data if d['label'] == 'POS'])))
    if len([d for d in data if d['label'] == 'POS']) > 0:
        print_func("Avg POS Seq Length: {:5.3f}".format(sum(len(d['varg_seq']) for d in data if d['label'] == 'POS') / len([d for d in data if d['label'] == 'POS'])))
    else:
        print_func("Avg POS Seq Length: 0.")
    print_func("Num NEG Seq: {:d}".format(len([d for d in data if d['label'] == 'NEG'])))
    if len([d for d in data if d['label'] == 'NEG']) > 0:
        print_func("Avg NEG Seq Length: {:5.3f}".format(sum(len(d['varg_seq']) for d in data if d['label'] == 'NEG') / len([d for d in data if d['label'] == 'NEG'])))
    else:
        print_func("Avg NEG Seq Length: 0.")


def print_stat_chainlen(data, print_func=print):
    ls = [len(d['varg_seq']) for d in data]
    for i in range(max(ls)+1):
        print_func("length {:d}: {:5.3f}%".format(i, (sum([l == i for l in ls]) / len(ls)) * 100))
