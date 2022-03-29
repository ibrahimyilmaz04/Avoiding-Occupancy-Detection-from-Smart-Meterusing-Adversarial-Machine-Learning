from hashlib import sha256
import json
import time
import csv
import os

class Block:
	def __init__(self, index, transactions, timestamp, previous_hash):
		self.index = index
		self.transactions = transactions
		self.timestamp = timestamp
		self.previous_hash = previous_hash
		self.nonce = 0

	def compute_hash(self):
		block_string = json.dumps(self.__dict__, sort_keys=True)
		return sha256(block_string.encode()).hexdigest()

print ('first')
class Blockchain:
	difficulty = 1

	def __init__(self):
		self.unconfirmed_transactions = []
		self.chain = []
		self.create_genesis_block()

	def create_genesis_block(self):
		genesis_block = Block(0, [], time.time(), "0")
		genesis_block.hash = genesis_block.compute_hash()
		self.chain.append(genesis_block)

	@property
	def last_block(self):
		return self.chain[-1]

	def add_block(self, block, proof):
		previous_hash = self.last_block.hash

		if previous_hash != block.previous_hash:
			return False

		if not self.is_valid_proof(block, proof):
			return False

		block.hash = proof
		self.chain.append(block)
		return True

	def is_valid_proof(self, block, block_hash):
		return (block_hash.startswith('0' * Blockchain.difficulty) and
				block_hash == block.compute_hash())

	def proof_of_work(self, block):
		block.nonce = 0

		computed_hash = block.compute_hash()
		while not computed_hash.startswith('0' * Blockchain.difficulty):
			block.nonce += 1
			computed_hash = block.compute_hash()

		return computed_hash

	def add_new_transaction(self, transaction):
		self.unconfirmed_transactions.append(transaction)

	def mine(self):
		if not self.unconfirmed_transactions:
			return False

		last_block = self.last_block

		new_block = Block(index=last_block.index + 1,
						  transactions=self.unconfirmed_transactions,
						  timestamp=time.time(),
						  previous_hash=last_block.hash)

		proof = self.proof_of_work(new_block)
		self.add_block(new_block, proof)

		self.unconfirmed_transactions = []
		return new_block.index

kk = Blockchain()
tx_data = {}
time_list = []
kkk_time = time.time()
for n in range(5):
	t1 = time.time()
	root = f"/home/tntech.edu/iyilmaz42/occupancy/occupancy_dataset/home{n+1}"
	for path, subdirs, files in os.walk(root):
		for name in files:
			k = ''
			print(os.path.join(path, name).replace("\\","/"))
			if "plugs" in os.path.join(path, name):
				k = os.path.join(path, name)[os.path.join(path, name).index("plugs"):-4]
			elif "summer" in os.path.join(path, name):
				k = "Summer occupancy"
			elif "winter" in os.path.join(path, name):
				k = "Winter occupancy"
			with open(os.path.join(path, name), mode='r') as infile:
				try:
					reader = csv.reader(infile)
					data = [i for i in reader]
					tx_data[k] = data
					# tx_data[k] = "kk"
				except Exception as e:
					continue
	kk.add_new_transaction(tx_data)
	kk.mine()
	time_list.append(time.time()-t1)
for i in kk.chain:
	print(i.__dict__)
for k,j in enumerate(time_list):
	print(f'execution time for home {k+1} in seconds is:',j)
print('execution time for whole operation in seconds is:',(time.time()-kkk_time))