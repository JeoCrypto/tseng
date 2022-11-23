class SearchEngine:
    def __init__(self, db):
        self.db = db

    def add_corpus(self, user_id, text):
        words = self.tokenize(text)
        word_ids = set()
        for word in words:
            word_id = self.hash_word(word)
            word_ids.add(word_id)
            self.db.execute(
                "insert into inverted_index (word_id, user_id) values (%s, %s)", (word_id, user_id))
        for word_id in word_ids:
            self.db.execute("update inverted_index set count = count + 1 where word_id = %s and user_id = %s",
                            (word_id, user_id))

    def search(self, query):
        words = self.tokenize(query)
        word_ids = list(map(self.hash_word, words))
        if word_ids:
            user_ids = self.db.query(
                "select user_id from inverted_index where word_id in ({}) group by user_id having sum(count) = {}".format(
                    ",".join(["%s"] * len(word_ids)), len(word_ids)), word_ids)
            return user_ids
        else:
            return []

    def hash_word(self, word):
        if len(word) > 20:
            return hash(word[:20])
        else:
            return hash(word)

    def tokenize(self, text):
        return text.split()
    