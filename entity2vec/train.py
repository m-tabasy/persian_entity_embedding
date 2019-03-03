import time
import torch

from entity2vec import minibatch
from basic import commons

config = commons.get_config()

model_path = config['path']['model']

# -------- globals --------

epochs_passed = 0
loss_history = []


# -------- functions --------

def update_checkpoint(loss):
	global loss_history
	loss_history.append(loss)


def save_checkpoint(model, optimizer=None):
	global model_path, loss_history
	
	checkpoint = {
		'epoch': len(loss_history),
		'loss': loss_history[-1],
		'history': loss_history,
		'state': model.state_dict()
	}
	
	if optimizer is not None:
		checkpoint['optimizer'] = optimizer.state_dict()
	
	torch.save(checkpoint, model_path)


def load_checkpoint(model, optimizer=None):
	global model_path, epochs_passed, loss_history
	
	checkpoint = torch.load(model_path)
	epochs_passed = checkpoint['epoch']
	loss_history = checkpoint['history']
	model.load_state_dict(checkpoint['state'])
	
	if optimizer is not None:
		model.load_state_dict(checkpoint['optimizer'])


def default_stop_condition(loss_history):
	if len(loss_history) < 3:
		return False
	
	pre_improve = loss_history[-3] - loss_history[-2]
	current_improve = loss_history[-2] - loss_history[-1]
	
	return pre_improve < 0 and current_improve < 0


def train_entity_vectors(network, loss_func, optimizer, epochs=1,
                         stop_condition=default_stop_condition, do_before_epoch=None, do_after_epoch=None):
	global epochs_passed, loss_history
	
	for epoch in range(epochs):
		
		if do_before_epoch is not None:
			do_before_epoch(loss_history)
		
		print(f'-- epoch #{epochs_passed + 1}:')
		
		epoch_loss = 0
		batch_index = 0
		
		for inputs, targets in minibatch.gen_minibatch():
			def closure():
				optimizer.zero_grad()
				output = network(inputs)
				loss = loss_func(output, targets)
				loss.backward()
				return loss
			
			_t = time.time()  #
			
			optimizer.step(closure)
			epoch_loss += closure()
			batch_index += 1
			
			minibatch._times[7] += time.time() - _t  #
		
		epochs_passed += 1
		
		update_checkpoint(epoch_loss / batch_index)
		
		print(f'\n-- loss: {epoch_loss}')
		
		if len(loss_history) > 1:
			print(f'-- improvement: {loss_history[-2] - loss_history[-1]}')
		
		if epochs_passed % 2 == 0 or stop_condition(loss_history):
			print(f'-- saving model!')
			save_checkpoint(network, optimizer)
		
		if do_after_epoch is not None:
			do_after_epoch(loss_history)
		
		if stop_condition(loss_history):
			print(f'-- stop condition is satisfied. stopping!!')
			break

