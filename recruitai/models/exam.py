from extensions import db


class ExamQuestion(db.Model):
    __tablename__ = 'exam_questions'
    id             = db.Column(db.Integer, primary_key=True, autoincrement=True)
    job_role       = db.Column(db.String(100), nullable=False, index=True)
    question_text  = db.Column(db.Text,        nullable=False)
    option_a       = db.Column(db.String(500), nullable=False)
    option_b       = db.Column(db.String(500), nullable=False)
    option_c       = db.Column(db.String(500), nullable=False)
    option_d       = db.Column(db.String(500), nullable=False)
    correct_answer = db.Column(db.String(1),   nullable=False)
    difficulty     = db.Column(db.String(20),  default='medium')
    category       = db.Column(db.String(100))

    def to_dict(self, include_answer=False):
        d = {
            'id': self.id, 'job_role': self.job_role,
            'question_text': self.question_text,
            'options': {'a': self.option_a, 'b': self.option_b,
                        'c': self.option_c, 'd': self.option_d},
            'difficulty': self.difficulty, 'category': self.category,
        }
        if include_answer:
            d['correct_answer'] = self.correct_answer
        return d
