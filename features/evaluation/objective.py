import nltk
import numpy as np
import re
from nltk.corpus import wordnet as wn

class ObjectiveTest:

    def __init__(self, summary, noOfQues):
        self.summary = summary
        self.noOfQues = noOfQues

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        trivial_sentences = list()
        for sent in sentences:
            trivial = self.identify_trivial_sentences(sent)
            if trivial:
                trivial_sentences.append(trivial)
        return trivial_sentences

    def identify_trivial_sentences(self, sentence):
        words = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(words)

        if tags[0][1] == "RB" or len(words) < 4:
            return None

        grammar = r"""
                   CHUNK: {<NN>+<IN|DT>*<NN>+}
                          {<NN>+<IN|DT>*<NNP>+}
                          {<NNP>+<NNS>*}
               """
        chunker = nltk.RegexpParser(grammar)
        tree = chunker.parse(tags)

        noun_phrases = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'CHUNK'):
            phrase = " ".join(word for word, tag in subtree.leaves())
            noun_phrases.append(phrase)

        if not noun_phrases:
            return None

        # Select the longest noun phrase to replace
        replace_nouns = max(noun_phrases, key=len).split()

        val = min(len(i) for i in replace_nouns)

        trivial = {
            "Answer": " ".join(replace_nouns),
            "Key": val
        }

        if len(replace_nouns) == 1:
            trivial["Similar"] = self.answer_options(replace_nouns[0])
        else:
            trivial["Similar"] = []

        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = ("__________ " * len(replace_nouns)).strip()
        expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
        sentence = expression.sub(blanks_phrase, sentence, count=1)
        trivial["Question"] = sentence
        return trivial

    @staticmethod
    def answer_options(word):
        synsets = wn.synsets(word, pos="n")
        if not synsets:
            return []

        hypernym = synsets[0].hypernyms()[0] if synsets[0].hypernyms() else None
        if not hypernym:
            return []

        hyponyms = hypernym.hyponyms()
        similar_words = [hyponym.lemmas()[0].name().replace("_", " ") for hyponym in hyponyms if hyponym.lemmas()[0].name() != word]
        return similar_words[:8]

    def generate_test(self):
        trivial_pair = self.get_trivial_sentences()
        if not trivial_pair:
            raise ValueError("No trivial sentences found in the input text.")

        question_answer = [que_ans_dict for que_ans_dict in trivial_pair if que_ans_dict]

        if not question_answer:
            raise ValueError("No valid questions could be generated from the input text.")

        question = []
        answer = []

        while len(question) < int(self.noOfQues):
            rand_num = np.random.randint(0, len(question_answer))
            if question_answer[rand_num]["Question"] not in question:
                question.append(question_answer[rand_num]["Question"])
                answer.append(question_answer[rand_num]["Answer"])

        return question, answer

