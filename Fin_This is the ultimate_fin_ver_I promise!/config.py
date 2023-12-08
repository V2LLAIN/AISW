import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=2e-4)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
parser.add_argument('--epoch_num', dest='epoch_num', type=int, default=16)
parser.add_argument('--MAX_LEN', dest='MAX_LEN', type=int, default=512)
parser.add_argument('--seed', dest='seed', type=int, default=100)
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.01)

parser.add_argument('--padding', dest='padding', type=bool, default=True)
parser.add_argument('--truncation', dest='truncation', type=bool, default=True)
parser.add_argument('--return_token_type_ids', dest='return_token_type_ids', type=bool, default=True)

parser.add_argument('--model_name', dest='model_name', default='bert-base-uncased')

args = parser.parse_args()
