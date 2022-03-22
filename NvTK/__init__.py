from .Modules import *

from .Model import *
from .Model import Extention

from .Trainer import Trainer
from .Evaluator import *
from .Explainer import *

def set_random_seed(random_seed = 12):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.seed = random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) 
    torch.cuda.manual_seed_all(random_seed)


def set_torch_benchmark():
    # set torch benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
