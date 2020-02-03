from enum import Enum
import numpy as np
import math

class VAD:
    def __init__(self, v, a, d):
        self.v = v
        self.a = a
        self.d = d

    def __add__(self, other):
        return VAD(self.v + other.v, self.a + other.a, self.d + other.d)

    def __radd__(self, other):
        return VAD(self.v + other, self.a + other, self.d + other)

    def __sub__(self, other):
        return VAD(self.v - other.v, self.a - other.a, self.d - other.d)

    def __rsub__(self, other):
        return VAD(self.v - other, self.a - other, self.d - other)

    def __div__(self, b):
        return self.__truediv__(b)

    def __truediv__(self, b):
        return VAD(self.v / b, self.a / b, self.d / b)

    def __mul__(self, scaling):
        return VAD(self.v * scaling, self.a * scaling, self.d * scaling)

    def __rmul__(self, scaling):
        return VAD(self.v * scaling, self.a * scaling, self.d * scaling)

    def __str__(self):
        return "V: {0}, A: {1}, D: {2}".format(self.v, self.a, self.d)

    def __repr__(self):
        return self.__str__()

    def dist(self, other):
        return np.linalg.norm(
            np.array([self.v, self.a, self.d]) - np.array([other.v, other.a, other.d])
            )

    def closest(self, emotionSet):
        keys = list(emotionSet.keys())
        values = list(emotionSet.values())
        distances = [self.dist(emot) for emot in values]

        return keys[np.argmin(distances)]

    def topKClosest(self, emotionSet, k=5):
        keys = list(emotionSet.keys())
        values = list(emotionSet.values())
        dists = [self.dist(emot) for emot in values]
        distsZip = list(zip(keys, dists))
        distsZipSorted = sorted(distsZip, key=lambda x: x[1])
        return distsZipSorted[:k]

Ekman = {
    'SADNESS': VAD(-0.63, -0.27, -0.33),
    'FEAR': VAD(-0.64, 0.60, -0.43),
    'DISGUST': VAD(-0.60, 0.35, 0.11),
    'ANGER': VAD(-0.51, 0.59, 0.25),
    'SURPRISE': VAD(0.40, 0.67, -0.13),
    'JOY': VAD(0.76, 0.48, 0.35)
}

Crowdflower = {
    'BOREDOM': VAD(-0.65, -0.62, -0.33),
    'ENTHUSIASM': VAD(0.62, 0.75, 0.38),
    'FUN': VAD(0.77, 0.44, 0.42),
    'HAPPINESS': VAD(0.81, 0.51, 0.46),
    'HATE': VAD(-0.56, 0.59, 0.13),
    'LOVE': VAD(0.87, 0.54, -0.18),
    # 'NEUTRAL': VAD(0, 0, 0),
    'RELIEF': VAD(0.68, -0.46, 0.6),
    'WORRY': VAD(-0.63, 0.16, -0.40)
}

Custom = {
    'LOVE': VAD(0.87, 0.54, -0.18),
    # 'HATE': VAD(-0.56, 0.59, 0.13),
    'JOY': VAD(0.76, 0.48, 0.35),
    'SADNESS': VAD(-0.63, -0.27, -0.33),
    # 'DEFETEAD': VAD(-0.61, 0.06, -0.32),
    # 'EXCITED': VAD(0.62, 0.75, 0.38),
    # 'BLASE': VAD(-0.29, -0.51, -0.16),
    'RELAXED': VAD(0.688, -0.46, 0.06),
    'TRIUMPHANT': VAD(0.69, 0.57, 0.63),
    # 'CRUSHED': VAD(-0.69, 0.03, -0.50),
    # 'AWED': VAD(0.18, 0.40, -0.38),
    'NEUTRAL': VAD(-0.37, -0.26, -0.14)
}

Clustered = {
    '0': VAD(-0.55494132, -0.20019732, -0.53936147),
    '1': VAD(0.10463187,  0.25858045,  0.31068011),
    '2': VAD(-0.18524508,  0.02764591, -0.1623381),
    # '3': VAD(0.43938088, -0.33324177, -0.19668701),
    # '4': VAD(-0.54695341,  0.61840083,  0.18222976),
    # '5': VAD(0.5073737 , -0.18800611,  0.37610334),
    # '6': VAD(-0.02106802, -0.42497992, -0.34629592),
    '7': VAD(0.63850704,  0.35651016,  0.48128517),
    # '8': VAD(-0.66427494,  0.38769898, -0.32192147),
    '9': VAD(0.09950419, -0.21007509,  0.09023259)
}

SongClustered = {
    '1': VAD(-0.19761926, -0.21134848, -0.31661923),
    '2': VAD(0.43598551, -0.19669591,  0.1349824),
    '3': VAD(0.07152347,  0.09967   ,  0.01894801),
    '4': VAD(0.63580989,  0.10286462,  0.32711383),
    '5': VAD(0.19423887, -0.31599715, -0.12945419),
    '6': VAD(-0.5161391 ,  0.37670959, -0.18224283)
}

Final = {
    'Relaxed': VAD(0.68, -0.46, 0.06),
    'Solemn': VAD(0.03, -0.32, -0.11),
    'Joyful': VAD(0.76, 0.48, 0.35),
    'Startled': VAD(-0.09, 0.65, -0.33),
    'Upset': VAD(-0.63, 0.30, -0.24),
    'Sad': VAD(-0.63, -0.27, -0.33),
}

ThreeFactor = {
    'Bold': VAD(0.44, 0.61, 0.66),
    'Useful': VAD(0.70, 0.44, 0.47),
    'Mighty': VAD(0.48, 0.51, 0.69),
    'Kind': VAD(0.73, 0.19, 0.57),
    'Self-satisfied': VAD(0.86, 0.20, 0.62),
    'Admired': VAD(0.81, 0.44, 0.51),
    'Proud': VAD(0.77, 0.38, 0.65),
    'Interested': VAD(0.64, 0.51, 0.17),
    'Arrogant': VAD(0.00, 0.34, 0.48),
    'Inspired': VAD(0.71, 0.63, 0.34),
    'Excited': VAD(0.62, 0.75, 0.38),
    'Influentian': VAD(0.68, 0.40, 0.75),
    'Aggresive': VAD(0.41, 0.63, 0.62),
    'Strong': VAD(0.58, 0.48, 0.62),
    'Dignified': VAD(0.55, 0.22, 0.61),
    'Powerful': VAD(0.54, 0.45, 0.73),
    'Elated': VAD(0.50, 0.42, 0.23),
    'Hopeful': VAD(0.51, 0.23, 0.14),
    'Triumphant': VAD(0.69, 0.57, 0.63),
    'Joyful': VAD(0.76, 0.48, 0.35),
    'Capable': VAD(0.70, 0.28, 0.61),
    'Lucky': VAD(0.71, 0.48, 0.37),
    'Responsible': VAD(0.35, 0.38, 0.49),
    'Friendly': VAD(0.69, 0.35, 0.30),
    'Masterful': VAD(0.58, 0.44, 0.69),
    'Free': VAD(0.81, 0.24, 0.46),
    'Devoted': VAD(0.49, 0.17, 0.10),
    'Domineering': VAD(0.23, 0.40, 0.58),
    'Aroused': VAD(0.24, 0.57, 0.22),
    'Concentrating': VAD(0.42, 0.28, 0.39),
    'Happy': VAD(0.81, 0.51, 0.46),
    'Egotistical': VAD(0.24, 0.32, 0.50),
    'Carefree': VAD(0.78, 0.25, 0.41),
    'Affectionate': VAD(0.64, 0.35, 0.24),
    'Vigorous': VAD(0.58, 0.61, 0.49),
    'Activated': VAD(0.42, 0.58, 0.38),
    'Alert': VAD(0.49, 0.57, 0.45),
    'Alone with responsibility (wtf?)': VAD(0.33, 0.34, 0.48),
    'Controlling': VAD(0.47, 0.34, 0.66),
    'Proud and lonely': VAD(0.01, 0.02, 0.26),
    'Enjoyment': VAD(0.77, 0.44, 0.42),
    'Serious': VAD(0.27, 0.24, 0.42),
    'Cooperative': VAD(0.39, 0.13, 0.03),
    'Thankful': VAD(0.61, 0.10, -0.13),
    'Respectful': VAD(0.38, 0.13, -0.08),
    'Appreciative': VAD(0.55, 0.07, -0.14),
    'Loved': VAD(0.87, 0.54, -0.18),
    'Grateful': VAD(0.64, 0.16, -0.18),
    'In love': VAD(0.82, 0.65, -0.05),
    'Anxious': VAD(0.01, 0.59, -0.15),
    'Impressed': VAD(0.41, 0.30, -0.32),
    'Surprised': VAD(0.40, 0.67, -0.13),
    'Sexually excited': VAD(0.58, 0.62, -0.01),
    'Wonder': VAD(0.27, 0.24, -0.17),
    'Fascinated': VAD(0.55, 0.51, -0.07),
    'Awed': VAD(0.18, 0.40, -0.38),
    'Overwhelmed': VAD(0.14, 0.45, -0.24),
    'Curious': VAD(0.22, 0.62, -0.01),
    'Relaxed': VAD(0.68, -0.46, 0.06),
    'Untroubled': VAD(0.79, -0.01, 0.33),
    'Modest': VAD(0.27, -0.06, 0.12),
    'Secure': VAD(0.74, -0.13, 0.03),
    'Nonchalant': VAD(0.07, -0.25, 0.11),
    'Aloof': VAD(0.16, -0.01, 0.25),
    'Leisurely': VAD(0.58, -0.32, 0.11),
    'Reserved': VAD(0.01, -0.19, 0.02),
    'Protected': VAD(0.60, -0.22, -0.42),
    'Consoled': VAD(0.29, -0.19, -0.28),
    'Quiet': VAD(0.19, -0.40, -0.04),
    'Sheltered': VAD(0.14, -0.36, -0.44),
    'Humble': VAD(0.23, -0.28, -0.27),
    'Solemn': VAD(0.03, -0.32, -0.11),
    'Reverent': VAD(0.31, -0.08, -0.29),
    'Astonished': VAD(0.16, 0.88, -0.15),
    'Disgusted': VAD(-0.60, 0.35, 0.11),
    'Insolent': VAD(-0.26, 0.21, 0.20),
    'Defiant': VAD(-0.16, 0.54, 0.32),
    'Hate': VAD(-0.56, 0.59, 0.13),
    'Hostile': VAD(-0.42, 0.53, 0.30),
    'Angry': VAD(-0.51, 0.59, 0.25),
    'Mildly annoyed': VAD(-0.28, 0.17, 0.04),
    'Enraged': VAD(-0.44, 0.72, 0.32),
    'Contempt': VAD(-0.23, 0.31, 0.18),
    'Selfish': VAD(-0.34, 0.09, 0.31),
    'Reprehensible': VAD(-0.09, 0.11, 0.06),
    'Contemptuous': VAD(-0.24, 0.31, 0.21),
    'Scornful': VAD(-0.35, 0.35, 0.29),
    'Suspicious': VAD(-0.25, 0.42, 0.11),
    'Skeptical': VAD(-0.22, 0.21, 0.03),
    'Burdened with responsibility': VAD(-0.08, 0.28, 0.19),
    'Cold anger': VAD(-0.43, 0.67, 0.34),
    'Hostile but controlled': VAD(-0.24, 0.42, 0.09),
    'Crushed': VAD(-0.69, 0.03, -0.5),
    'Frustrated': VAD(-0.64, 0.52, -0.35),
    'Distressed': VAD(-0.61, 0.28, -0.36),
    'Insecure': VAD(-0.57, 0.14, -0.21),
    'Humiliated': VAD(-0.63, 0.43, -0.38),
    'Hungry': VAD(-0.44, 0.14, -0.21),
    'Fearful': VAD(-0.64, 0.60, -0.43),
    'Terrified': VAD(-0.62, 0.82, -0.43),
    'Embattled': VAD(-0.37, 0.40, -0.02),
    'Helpess': VAD(-0.71, 0.42, -0.51),
    'Troubled': VAD(-0.63, 0.16, -0.40),
    'Startled': VAD(-0.09, 0.65, -0.33),
    'Anguished': VAD(-0.50, 0.08, -0.20),
    'Shamed': VAD(-0.57, 0.01, -0.34),
    'Displeased': VAD(-0.55, 0.16, -0.05),
    'Embarrassed': VAD(-0.46, 0.54, -0.24),
    'Upset': VAD(-0.63, 0.30, -0.24),
    'Defetead': VAD(-0.61, 0.06, -0.32),
    'Pain': VAD(-0.58, 0.41, -0.34),
    'Quitely indignant': VAD(-0.28, 0.04, -0.16),
    'Repentant': VAD(-0.06, 0.06, -0.12),
    'Sinful': VAD(-0.30, 0.22, -0.01),
    'Shy': VAD(-0.15, 0.06, -0.34),
    'Guilty': VAD(-0.57, 0.28, -0.34),
    'Weary with responsibility': VAD(-0.27, 0.02, -0.01),
    'Angry but detached': VAD(-0.42, 0.28, -0.03),
    'Confused': VAD(-0.53, 0.27, -0.32),
    'Dissatisfied': VAD(-0.50, 0.05, 0.13),
    'Refretful': VAD(-0.52, 0.02, -0.21),
    'Tense': VAD(-0.33, 0.58, -0.11),
    'Disdainful': VAD(-0.32, -0.11, 0.05),
    'Depressed': VAD(-0.72, -0.29, -0.41),
    'Despairing': VAD(-0.72, -0.16, -0.38),
    'Lonely': VAD(-0.66, -0.43, -0.32),
    'Meek': VAD(-0.19, -0.25, -0.41),
    'Burdened': VAD(-0.66, -0.03, -0.26),
    'Timid': VAD(-0.15, -0.12, -0.47),
    'Bored': VAD(-0.65, -0.62, -0.33),
    'Feeble': VAD(-0.42, -0.20, -0.46),
    'Nauseated': VAD(-0.61, -0.01, -0.36),
    'Inhibited': VAD(-0.54, -0.04, -0.41),
    'Fatigued': VAD(-0.18, -0.57, -0.29),
    'Rejected': VAD(-0.62, -0.01, -0.33),
    'Subdued': VAD(-0.17, -0.26, -0.18),
    'Impotent': VAD(-0.53, -0.13, -0.29),
    'Ennui': VAD(-0.45, -0.43, -0.17),
    'Blase': VAD(-0.29, -0.51, -0.16),
    'Haughty and lonely': VAD(-0.47, -0.24, -0.13),
    'Listless': VAD(-0.45, -0.59, -0.24),
    'Deactivated': VAD(-0.46, -0.43, -0.46),
    'Weary': VAD(-0.18, -0.33, -0.24),
    'Snobbish and lonely': VAD(-0.62, -0.19, -0.14),
    'Uninterested': VAD(-0.47, -0.50, -0.08),
    'Detached': VAD(-0.37, -0.26, -0.14),
    'Discontented': VAD(-0.53, -0.16, -0.26),
    'Discouraged': VAD(-0.61, -0.15, -0.29),
    'Sad': VAD(-0.63, -0.27, -0.33)
}
