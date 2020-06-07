import itertools
import multiprocessing
import time
import spam_classifier


options = [
    "CONTENT_FROM_BODY",
    "STEM_WORDS",
    "REMOVE_STOP_WORDS",
    "MIN_COUNT",
    "SPECIAL_TOKENS",
    "NLTK_TOKENIZE",
    "ADD_FROM",
    "ADD_URL",
    "ADD_RECEIVED",
]


def combinations(some_list):
    all_combinations = []
    for i in range(len(some_list) + 1):
        all_combinations += list(itertools.combinations(some_list, i))
    return all_combinations


if __name__ == "__main__":
    starttime = time.time()
    processes = []
    for active_options in combinations(options):
        settings_options = {o: o in active_options for o in options}

        p = multiprocessing.Process(
            target=spam_classifier.train_and_test_model,
            args=(r"./emails/*/*", settings_options),
        )
        processes.append(p)
        p.start()

    for proccess in processes:
        proccess.join()

    print("That took {} seconds".format(time.time() - starttime))
