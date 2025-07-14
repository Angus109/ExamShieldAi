import re
import numpy as np
import nltk


class SubjectiveTest:

    def __init__(self, filepath, noOfQues):

        self.question_pattern = [
            "Explain in detail ",
            "Define ",
            "Write a short note on ",
            "What do you mean by "
        ]
        self.grammar = r"""
                   CHUNK: {<NN>+<IN|DT>*<NN>+}
                   {<NN>+<IN|DT>*<NNP>+}
                   {<NNP>+<NNS>*}
               """
        self.summary = filepath
        self.noOfQues = noOfQues

    @staticmethod
    def word_tokenizer(sequence):
        word_tokens = list()
        for sent in nltk.sent_tokenize(sequence):
            for w in nltk.word_tokenize(sent):
                word_tokens.append(w)
        return word_tokens

    def generate_test(self):
        sentences = nltk.sent_tokenize(self.summary)
        cp = nltk.RegexpParser(self.grammar)
        question_answer_dict = dict()
        for sentence in sentences:
            tagged_words = nltk.pos_tag(nltk.word_tokenize(sentence))
            tree = cp.parse(tagged_words)
            for subtree in tree.subtrees():
                if subtree.label() == "CHUNK":
                    temp = ""
                    for sub in subtree:
                        temp += sub[0]
                        temp += " "
                    temp = temp.strip()
                    # Consider if uppercase conversion is desired
                    # temp = temp.upper()
                    if temp not in question_answer_dict:
                        question_answer_dict[temp] = sentence

        keyword_list = list(question_answer_dict.keys())

        if not keyword_list:  # Handle empty keyword list
            # Return an error message or default questions here
            return None, None

        question_answer = list()
        for _ in range(int(self.noOfQues)):
            while len(question_answer) < int(self.noOfQues):
                rand_num = np.random.randint(0, len(keyword_list))
                selected_key = keyword_list[rand_num]
                question = self.question_pattern[rand_num % 4] + selected_key + "."
                if question not in [q["Question"] for q in question_answer]:
                    question_answer.append({"Question": question, "Answer": question_answer_dict[selected_key]})
                    break  # Exit inner loop after adding a unique question

        que = [q["Question"] for q in question_answer]
        ans = [q["Answer"] for q in question_answer]
        return que, ans
