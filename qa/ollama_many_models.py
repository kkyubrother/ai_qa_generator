from abc import ABC, abstractmethod
from typing import List, Optional
import aiohttp
import json
import math


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    def dot_product(v1: List[float], v2: List[float]) -> float:
        return sum(x * y for x, y in zip(v1, v2))

    def magnitude(vec: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    dot = dot_product(vec1, vec2)
    mag1 = magnitude(vec1)
    mag2 = magnitude(vec2)
    return dot / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0


class Generator(ABC):
    def __init__(self, settings):
        self.settings = settings

    @abstractmethod
    async def generate_quiz(self, contents: List[str]) -> Optional[str]:
        pass

    @abstractmethod
    async def short_or_long_answer_similarity(self, user_answer: str, answer: str) -> float:
        pass

    def system_prompt(self) -> str:
        true_false_format = (
            '"question": The question\n"answer": A boolean representing the answer\n'
        )
        multiple_choice_format = (
            '"question": The question\n"options": An array of 4 to 26 strings representing the choices\n'
            '"answer": The number corresponding to the index of the correct answer in the options array\n'
        )
        select_all_that_apply_format = (
            '"question": The question\n"options": An array of 4 to 26 strings representing the choices\n'
            '"answer": An array of numbers corresponding to the indexes of the correct answers in the options array\n'
        )
        fill_in_the_blank_format = (
            '"question": The question with 1 to 10 blanks, which must be represented by `____` (backticks included)\n'
            '"answer": An array of strings corresponding to the blanks in the question\n'
        )
        matching_format = (
            '"question": The question\n'
            '"answer": An array of 3 to 13 objects, each containing a leftOption property (a string that needs to be matched) '
            'and a rightOption property (a string that matches the leftOption)\n'
        )
        short_or_long_answer_format = (
            '"question": The question\n"answer": The answer\n'
        )

        question_formats = [
            {"generate": self.settings.generate_true_false, "format": true_false_format, "type": "true or false"},
            {"generate": self.settings.generate_multiple_choice, "format": multiple_choice_format,
             "type": "multiple choice"},
            {"generate": self.settings.generate_select_all_that_apply, "format": select_all_that_apply_format,
             "type": "select all that apply"},
            {"generate": self.settings.generate_fill_in_the_blank, "format": fill_in_the_blank_format,
             "type": "fill in the blank"},
            {"generate": self.settings.generate_matching, "format": matching_format, "type": "matching"},
            {"generate": self.settings.generate_short_answer, "format": short_or_long_answer_format,
             "type": "short answer"},
            {"generate": self.settings.generate_long_answer, "format": short_or_long_answer_format,
             "type": "long answer"},
        ]

        active_formats = "\n".join(
            f'The JSON object representing {q["type"]} questions must have the following properties:\n{q["format"]}'
            for q in question_formats if q["generate"]
        )

        return (
                'You are an assistant specialized in generating exam-style questions and answers. '
                'Your response must only be a JSON object with the following property:\n'
                '"questions": An array of JSON objects, where each JSON object represents a question and answer pair. '
                'Each question type has a different JSON object format.\n\n'
                f'{active_formats}\n'
                f'For example, if I ask you to generate {self.system_prompt_questions()}, the structure of your response should look like this:\n'
                f'{self.example_response()}' +
                (f'\n\n{self.generation_language()}' if self.settings.language != "English" else "")
        )

    def user_prompt(self, contents: List[str]) -> str:
        return (
                f'Generate {self.user_prompt_questions()} about the following text:\n' + "".join(contents) +
                '\n\nIf the above text contains LaTeX, you should use $...$ (inline math mode) for mathematical symbols. '
                'The overall focus should be on assessing understanding and critical thinking.'
        )

    def system_prompt_questions(self) -> str:
        question_types = [
            {"generate": self.settings.generate_true_false, "singular": "true or false question"},
            {"generate": self.settings.generate_multiple_choice, "singular": "multiple choice question"},
            {"generate": self.settings.generate_select_all_that_apply, "singular": "select all that apply question"},
            {"generate": self.settings.generate_fill_in_the_blank, "singular": "fill in the blank question"},
            {"generate": self.settings.generate_matching, "singular": "matching question"},
            {"generate": self.settings.generate_short_answer, "singular": "short answer question"},
            {"generate": self.settings.generate_long_answer, "singular": "long answer question"},
        ]

        active_question_types = [q["singular"] for q in question_types if q["generate"]]
        if len(active_question_types) == 1:
            return active_question_types[0]
        elif len(active_question_types) == 2:
            return " and ".join(active_question_types)
        else:
            last_part = active_question_types.pop()
            return f'{", ".join(active_question_types)}, and {last_part}'

    def user_prompt_questions(self) -> str:
        question_types = [
            {"generate": self.settings.generate_true_false, "count": self.settings.number_of_true_false,
             "singular": "true or false question", "plural": "true or false questions"},
            {"generate": self.settings.generate_multiple_choice, "count": self.settings.number_of_multiple_choice,
             "singular": "multiple choice question", "plural": "multiple choice questions"},
            {"generate": self.settings.generate_select_all_that_apply,
             "count": self.settings.number_of_select_all_that_apply, "singular": "select all that apply question",
             "plural": "select all that apply questions"},
            {"generate": self.settings.generate_fill_in_the_blank, "count": self.settings.number_of_fill_in_the_blank,
             "singular": "fill in the blank question", "plural": "fill in the blank questions"},
            {"generate": self.settings.generate_matching, "count": self.settings.number_of_matching,
             "singular": "matching question", "plural": "matching questions"},
            {"generate": self.settings.generate_short_answer, "count": self.settings.number_of_short_answer,
             "singular": "short answer question", "plural": "short answer questions"},
            {"generate": self.settings.generate_long_answer, "count": self.settings.number_of_long_answer,
             "singular": "long answer question", "plural": "long answer questions"},
        ]

        active_question_types = [
            f'{q["count"]} {q["plural"] if q["count"] > 1 else q["singular"]}'
            for q in question_types if q["generate"]
        ]
        if len(active_question_types) == 1:
            return active_question_types[0]
        elif len(active_question_types) == 2:
            return " and ".join(active_question_types)
        else:
            last_part = active_question_types.pop()
            return f'{", ".join(active_question_types)}, and {last_part}'

    def example_response(self) -> str:
        true_false_example = '{"question": "HTML is a programming language.", "answer": false}'
        multiple_choice_example = '{"question": "Which of the following is the correct translation of house in Spanish?", "options": ["Casa", "Maison", "Haus", "Huis"], "answer": 0}'
        select_all_that_apply_example = '{"question": "Which of the following are elements on the periodic table?", "options": ["Oxygen", "Water", "Hydrogen", "Salt", "Carbon"], "answer": [0, 2, 4]}'
        fill_in_the_blank_example = '{"question": "The Battle of `____` was fought in `____`.", "answer": ["Gettysburg", "1863"]}'
        matching_example = '{"question": "Match the medical term to its definition.", "answer": [{"leftOption": "Hypertension", "rightOption": "High blood pressure"}, {"leftOption": "Bradycardia", "rightOption": "Slow heart rate"}, {"leftOption": "Tachycardia", "rightOption": "Fast heart rate"}, {"leftOption": "Hypotension", "rightOption": "Low blood pressure"}]}'
        short_answer_example = '{"question": "Who was the first President of the United States and what is he commonly known for?", "answer": "George Washington was the first President of the United States and is commonly known for leading the American Revolutionary War and serving two terms as president."}'
        long_answer_example = '{"question": "Explain the difference between a stock and a bond, and discuss the risks and potential rewards associated with each investment type.", "answer": "A stock represents ownership in a company and a claim on part of its profits. The potential rewards include dividends and capital gains if the company\'s value increases, but the risks include the possibility of losing the entire investment if the company fails. A bond is a loan made to a company or government, which pays interest over time and returns the principal at maturity. Bonds are generally considered less risky than stocks, as they provide regular interest payments and the return of principal, but they offer lower potential returns."}'

        examples = [
            {"generate": self.settings.generate_true_false, "example": true_false_example},
            {"generate": self.settings.generate_multiple_choice, "example": multiple_choice_example},
            {"generate": self.settings.generate_select_all_that_apply, "example": select_all_that_apply_example},
            {"generate": self.settings.generate_fill_in_the_blank, "example": fill_in_the_blank_example},
            {"generate": self.settings.generate_matching, "example": matching_example},
            {"generate": self.settings.generate_short_answer, "example": short_answer_example},
            {"generate": self.settings.generate_long_answer, "example": long_answer_example},
        ]

        active_examples = ", ".join(e["example"] for e in examples if e["generate"])

        return f'{{"questions": [{active_examples}]}}'

    def generation_language(self) -> str:
        return (
            f'The generated questions and answers must be in {self.settings.language}. However, your '
            'response must still follow the JSON format provided above. This means that while the values should '
            f'be in {self.settings.language}, the keys must be the exact same as given above, in English.'
        )


class OllamaGenerator(Generator):
    def __init__(self, settings):
        super().__init__(settings)
        self.ollama_base_url = self.settings.ollama_base_url

    async def generate_quiz(self, contents: List[str]) -> Optional[str]:
        print(self.system_prompt())
        print(self.user_prompt(contents))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.ollama_base_url}/generate",
                        json={
                            "model": self.settings.ollama_text_gen_model,
                            "system": self.system_prompt(),
                            "prompt": self.user_prompt(contents),
                            "format": "json",
                            "stream": False,
                        },
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response")
                    else:
                        response_text = await response.text()
                        raise Exception(f"Failed to generate quiz: {response.status} - {response_text}")
        except Exception as e:
            raise Exception(str(e))

    async def short_or_long_answer_similarity(self, user_answer: str, answer: str) -> float:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.ollama_base_url}/embed",
                        json={
                            "model": self.settings.ollama_embedding_model,
                            "input": [user_answer, answer],
                        },
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embeddings = data.get("embeddings", [])
                        if len(embeddings) == 2:
                            return cosine_similarity(embeddings[0], embeddings[1])
                        else:
                            raise Exception("Unexpected number of embeddings returned")
                    else:
                        response_text = await response.text()
                        raise Exception(f"Failed to compute embeddings: {response.status} - {response_text}")
        except Exception as e:
            raise Exception(str(e))


# Example usage
if __name__ == "__main__":
    import asyncio

    class QuizSettings:
        generate_true_false = True
        generate_multiple_choice = True
        # aya-expanse
        generate_select_all_that_apply = False
        generate_fill_in_the_blank = True
        generate_matching = False
        generate_short_answer = False
        generate_long_answer = False
        ollama_base_url = "http://localhost:11434/api"

        # 지정된 형식을 지키지 않음.
        ollama_text_gen_model = "mistral"

        # 출력 깨짐
        # ollama_text_gen_model = "solar-pro"

        # 지문이 올바르지 않음
        # ollama_text_gen_model = "hermes3"

        # 문장에 특수기호가 너무 많이 첨가됨. 올바른 답도 아닌걸로 보임
        # ollama_text_gen_model = "phi3.5:3.8b-mini-instruct-fp16"

        # 뭔가 살짝 애매함
        # ollama_text_gen_model = "mistral-small"

        # 빈칸 문제는 좋아 보이는데, 나머지는 그닥...
        # ollama_text_gen_model = "gemma2"

        # 뭔가 잘 안됨, dense, moe 모두 안되는듯
        # ollama_text_gen_model = "granite3-moe"

        # 고장
        # ollama_text_gen_model = "llama3-8b-korean"

        # 한글 학습 자체가 안되어있음.
        # ollama_text_gen_model = "llama3.2"
        # ollama_text_gen_model = "llama3.2:1b"

        # 한글은 말하지만, 결국 헛소리함
        # ollama_text_gen_model = "llama3.2-3b-korean"

        # 같은 문제를 반복해서 출제는 하는데, 똑똑함
        # ollama_text_gen_model = "llama3.1-8b-darkidol"

        # 좀 멍청해짐
        # ollama_text_gen_model = "qwen2.5"

        # 구림
        # ollama_text_gen_model = "gemma2:27b-instruct-q2_K"

        # 개쩜
        # ollama_text_gen_model = "aya"
        # ollama_text_gen_model = "aya-expanse"
        # 아주 성능 좋음!
        # ollama_text_gen_model = "command-r:35b-08-2024-q2_K"
        # 괜찮아보임
        # ollama_text_gen_model = "qwen2.5:32b-instruct-q2_K"
        # 괜찮아보임
        # ollama_text_gen_model = "EEVE-Korean-10.8B"

        # 멍청함
        # ollama_text_gen_model = "phi3.5"

        # 한글이 없어짐...
        # ollama_text_gen_model = "deepseek-llm"

        # 에러
        # ollama_text_gen_model = "hf.co/OpenBuddy/openbuddy-llama3.2-1b-v23.1-131k-Q4_K_M-GGUF"

        ollama_embedding_model = "bge-m3"
        language = "Korean"
        number_of_true_false = 2
        number_of_multiple_choice = 2
        number_of_select_all_that_apply = 1
        number_of_fill_in_the_blank = 1
        number_of_matching = 1
        number_of_short_answer = 1
        number_of_long_answer = 1


    async def main():
        settings = QuizSettings()
        generator = OllamaGenerator(settings)
        contents = [
            "그러자 어른들은 내게 충고하길 엉뚱한 보아 뱀이나 그리지 말고 지리, 역사, 샘(계산)나 문법에 취미를 들여보래. 그리 하여 난 여섯 날에 화가의 꿈을 접어야 했지. 내 첫 그림과 두 번째 그림이 영 쓸모 없자 낙담하고 말았거든. 나의 이런 일 들에 대해 어른들은 전혀 관심도 없었지. 설명을 해대는 아인 피곤하다는 투였으니.",
            "그래서 다른 직업을 선택하게 된 거야. 그게 하늘을 나는 비행사지. 난 정말 전 세계를 날아다녔어. 그리 되니 지리학도 좀 도움이 되데. 난 중국이나 애리조나(미국의 주 이름)도 한눈에 첫 보면 알았지. 밤에 길을 잃었을 때도 지리학에 대한 앎 이 도움이 되었고 말이야.",
            "살아오며 다양한 사람들을 만났어, 대갠 심각한 어른들이었지. 난 그들 사이에 살아야 했으니깐. 그럼 난 그들에게 내 그림을 보여주었단다. 내 의견을 개진하지 않고 말이야.",
            "명석해 보이는 이들을 만날 때면 난 아끼며 보관해오던 내 첫 그림을 그분들께 보여주었지. 그들이 정말로 이해할까 알 고 싶었거든. 하지만 대답은 항상 이랬어. 모자군요. 그럼 난 보아 뱀 얘긴 꺼내지도 않았지 물론 숲 얘기나 별 얘기도 하지 않았어. 난 그들에게 내 자신을 맞추며, 기껏 '브리지'(카드 게임의 일종), 골프, 정치 그리고 술에 관한 얘기만 했을 뿐이 야. 그럼 어른들은 죄다 날 합리적인 사람이라 말하며 기뻐들 했지."]

        # Generate a quiz
        quiz = await generator.generate_quiz(contents)
        print("Generated Quiz:")
        # print(quiz)
        print(json.loads(quiz))

        # Calculate cosine similarity between two answers
        # user_answer = "The mitochondria provides energy to the cell."
        # correct_answer = "The mitochondria is the powerhouse of the cell."
        # similarity = await generator.short_or_long_answer_similarity(user_answer, correct_answer)
        # print("Cosine Similarity:", similarity)


    asyncio.run(main())
