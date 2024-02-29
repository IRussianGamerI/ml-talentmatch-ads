import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import GigaChat

import json


JSON_PATH = '../data/case_2_data_for_members.json'

chat = GigaChat(verify_ssl_certs=False)

experienceItem_ = """
Годы работы {starts}-{ends} на позиции {position}.
И описал свою деятельность как: {description}"""

with open(JSON_PATH) as f:
    data = json.load(f)

item = 1
for type_ in ['confirmed_resumes', 'failed_resumes']:
    for resume in data[item][type_]:
        exp_ = ''
        for i in resume['experienceItem']:
            exp_ += experienceItem_.format(
                starts=i['starts'],
                ends=i['ends'],
                position=i['position'],
                description=i['description']
            )

        messages = [
            SystemMessage(
                content="Посмотри на опыт человека как рекрутер. Суммаризуй опыт работы. Выдели ключевые достижения"
            ),
            HumanMessage(content=f"{exp_}"),
        ]

        summarized_exp = chat(messages).content

        messages = [
            SystemMessage(
                content="Посмотри на вакансию и опыт кандидата как рекрутер. Ты субъективен, и тебе нужно выставить оценку релевантности между вакансией и резюме по шкале от 0 до 14. От этого зависит твоя зарплата, максимальную оценку ты не ставишь почти никогда. В ответе отправь только число."
            ),
            HumanMessage(content=f"Описание вакансии: {data[item]['vacancy']['description']}. Опыт кандидата: {summarized_exp}. Год его рождения {resume['birth_date']}. Кандидат написал о себе: {resume['about']}."),
        ]

        print(type_, resume['uuid'], chat(messages).content)
