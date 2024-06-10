import random
from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import os 
import pickle
from game import Board
from game import IllegalAction, GameOver
from agent import nTupleNetwork

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def pong() -> dict:
    return {"data": "pong!"}

@app.post('/ai-move')
def ai_move(data: dict) -> int:
    '''
    @body: input [int, int, int, int] -> use it directly to model
    @return: It will return the direction in which the AI should move.
    '''
    # NOTE: [int, int, int, int]의 형태, model input에 사용
    # board_info = data["board"].split(' ')
    board=[]
    data_info = data["board"].split(' ')
    data = [int (i) for i in data_info]
    for idx in range(4):
        for i in range(4):
            board.append(data[3-idx]%16)
            data[3-idx]=data[3-idx]//16
    board=board[::-1]
    
    with open('./tmp/model.pkl', 'rb') as f:
        agent = pickle.load(f)
    
    b = Board(board)
    s = b.board
    a_best = agent.best_action(b.board)
    b.act(a_best)
    s_after = b.board
    if s==s_after:
        print("occured")
        a_best = random.randint(0, 3)
    res = (a_best+1)%4
    return res

def load_agent(path):
    return pickle.load(path.open("rb"))
if __name__ == "__main__":

    path = Path("tmp").joinpath("model.pkl")
    SAVE_PATH = os.path.abspath(os.path.dirname(__file__))

    with open('./tmp/model.pkl', 'rb') as f:
	    n_games, agent = pickle.load(f)