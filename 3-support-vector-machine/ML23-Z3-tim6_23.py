import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score

REVIEW_COL_NAME = "Review"
SENTIMENT_COL_NAME = "Sentiment"

def load_file_paths_from_cmd() -> (str, str):
    """
    :return: first -> file path to training data, second -> file path to test data

    If at least 2 system arguments are passed, they are returned assuming to be file
        paths to data, if less than 2 are given, defaults are returned
    """
    if len(sys.argv) > 2:
        train_directory_name = sys.argv[1]
        test_directory_name = sys.argv[2]
        return train_directory_name, test_directory_name
    return "train.tsv", "test_preview.tsv"

def remove_short_words(row: str):
    ALLOW = ["no", "ne", "ok", "top", "bad", "not", "naj", "raw", "los", "vrh", "suv", "dry", "loj", "brz", "sit", "lep"]
    return " ".join([token for token in row.split(" ") if not token or len(token) >= 4 or token in ALLOW])


def emoticon_decoder(data):
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace(":\(", " lose ")
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace(":\)", " odlicno ")
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace(": \)", " odlicno ")
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace(":-\)", " odlicno ")
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace(":d", " odlicno ")
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace(";d", " odlicno ")
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace(";d", " odlicno ")
    return data


def remove_non_alpha_chars(data):
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace('\u0131', '')
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace("[.,'()!?+-/*@#$%^&`0-9:;=_|{}><\n\"]", " ")
    return data


def transform_negs(row: str):
    NEGS = ["ne", "nije", "nisam", "nisu", "nismo", "nece", "necemo", "necete", "necu"]
    transform = False
    transformed = []
    for tk in row.split(" "):
        if transform:
            transform = False
            transformed.append("!" + tk)
        elif tk in NEGS:
            transform = True
            continue
        transformed.append(tk)
    return " ".join(transformed)


def compress_tokens_of_interest(row: str):
    TKS_OF_INTER = [
        "bajat", "besplatn", "bezobraz", "bezukus", "bezvez", "bljutav", "bozanstven", "disappoint", "dobr", "fantastic", "fenomenal",
        "gnjec", "gumen", "hladn", "hval", "ideal", "izuzet", "izvanred", "izvrs", "jeftin", "kasn", "kolicin", "komentar", "korekt",
        "ljubaz", "lose", "losi", "najbolj", "najbrz", "najgor", "nejestiv", "neslan", "neukus", "odlic", "odusev", "odvrat",
        "ohlad", "osrednj", "perfekt", "podgre", "pogres", "povolj", "prekuvan", "prepecen", "preporu", "preskup", "preukus", "prezadov",
        "pristoj", "profesional", "prosec", "prosut", "razocar", "savrsen", "sjaj", "solid", "svez", "uzas", "vrhu", "zabor", "zadovolj",
        "zagore"
    ]
    tokens = row.split(" ")
    for ind, tk in enumerate(tokens):
        for toi in TKS_OF_INTER:
            if toi in tk:
                tokens[ind] = toi if tk[0] != "!" else "!" + toi
                break
    return " ".join(tokens)


def transform_diacritical_mark(data):
    reg_to_val = {
        "č": "c",
        "ć": "c",
        "ž": "z",
        "š": "s",
        "đ": "dj",
        "sh": "s",
        "ch": "c"
    }
    return replacer(data, reg_to_val)


def other_stuff(data):
    reg_to_val = {
        "nikad vise": "nikadavise",
        "nikada vise": "nikadavise",
    }
    return replacer(data, reg_to_val)


def replacer(data, reg_to_val):
    for reg, val in reg_to_val.items():
        data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.replace(reg, val)
    return data


def preprocessing(data):
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].str.lower()
    data = emoticon_decoder(data)
    data = remove_non_alpha_chars(data)
    data = transform_diacritical_mark(data)
    data = other_stuff(data)
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].apply(remove_short_words)
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].apply(transform_negs)
    data[REVIEW_COL_NAME] = data[REVIEW_COL_NAME].apply(compress_tokens_of_interest)
    return data


def vectorize(rew_train, rew_test):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(rew_train), vectorizer.transform(rew_test)


def fit(rew_train, sent_train):
    classifier = SVC(kernel="rbf", gamma=0.62, C=1.4)
    classifier.fit(rew_train, sent_train)
    return classifier


def predict(classifier, rew_train):
    return classifier.predict(rew_train)


def main():
    train_data_file_path, test_data_file_path = load_file_paths_from_cmd()

    train_data = pd.read_csv(train_data_file_path, delimiter='\t')
    test_data = pd.read_csv(test_data_file_path, delimiter='\t')

    train_data = preprocessing(train_data)
    test_data = preprocessing(test_data)

    rew_train, rew_test = vectorize(train_data[REVIEW_COL_NAME].values, test_data[REVIEW_COL_NAME].values)
    sent_train, sent_test = train_data[SENTIMENT_COL_NAME].values, test_data[SENTIMENT_COL_NAME].values

    classifier = fit(rew_train, sent_train)

    print(f1_score(sent_test, predict(classifier, rew_test), average='micro'))


if __name__ == "__main__":
    main()
