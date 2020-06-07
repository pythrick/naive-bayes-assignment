
if __name__ == '__main__':

    match = {
        "accuracy": 0,
        "settings": []
    }
    with open("results.txt") as lines:
        for line in lines:
            accuracy = float(line[10:28])
            line = line.strip().split("Settings: ")
            settings = eval(line[1])
            settings = [k for k, v in settings.items() if v]
            if accuracy > match["accuracy"]:
                match["accuracy"] = accuracy
                match["settings"] = [settings]
            elif accuracy == match["accuracy"]:
                match["settings"].append(settings)
        print(match["accuracy"])
        for setting in match["settings"]:
            print(setting)
