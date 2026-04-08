import numpy as np   
class Relationship:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.key = tuple(sorted((a.id, b.id)))
        self.distance = np.linalg.norm(np.array(a.position) - np.array(b.position))
        self.married = False
        self.children = 0
        self.score = self.rel_score(a, b)
        self.friends = self.score >= 0.7

    def rel_score(self, a, b, alpha=2):
        similarity = np.dot(a.genome_vector_normalized, b.genome_vector_normalized)
        noise = np.random.normal(0, 0.15)
        distance_penalty = np.exp(-self.distance / (alpha * (a.radius + b.radius)))
        return np.clip(similarity * distance_penalty + noise, -1, 1)