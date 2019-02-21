import torch

from entity2vec import minibatch
from basic import commons

config = commons.get_config()

model_path = config['path']['model']

# -------- globals --------

epochs_passed = 0


# -------- functions --------

def save_model(network):
	global model_path
	torch.save(network, model_path)


def train_entity_vectors(network, loss_func, optimizer, epochs=1):

	global epochs_passed

	for epoch in range(epochs):

		print(f'-- starting {epochs_passed + 1}-th epoch!')

		epoch_loss = None

		for inputs, targets in minibatch.gen_minibatch():

			def closure():
				optimizer.zero_grad()
				output = network(inputs)
				loss = loss_func(output, targets)
				loss.backward()
				return loss

			optimizer.step(closure)

			epoch_loss = closure()

			print('.', end=' ', flush=True)

		epochs_passed += 1

		print(f'-- loss: {epoch_loss}\n')

		if epochs_passed % 2 == 0:
			print(f'-- saving model!')
			save_model(network)

