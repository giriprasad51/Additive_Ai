# Abstract

In this work, we introduce Additive AI, a novel framework for exploring and training deep learning models through structured architectural variation and gradient-level cooperation. We define a model family composed of binary compositional structures, where each + (addition) operation represents a binary decision to activate or skip a submodule, such as a subactivation layer or residual connection. This yields an exponentially large space of ( 2^n ) model architectures from a shared parameter pool, where n is the number of additive decisions. 

To systematically analyze this architecture space, we conduct a parallel evaluation across multiple GPUs (Exp 1), where each GPU executes a unique structure on the same dataset. This allows us to directly compare performance metrics and identify top-K high-performing structures. In Exp 2, we co-train these top-K models using a distributed gradient averaging scheme via ring AllReduce, enabling architectural diversity while maintaining a unified training signal. In Exp 3, we propose a form of structural regularization by randomly activating or deactivating + operations during training, akin to dropout at the architectural level. This stochastic gating encourages robust feature learning and reduces overfitting by preventing reliance on any single fixed path. Together, these methods introduce a flexible and efficient approach to model structure exploration, ensemble-style training, and architectural regularization, opening new directions in neural architecture search and dynamic model design.

In Exp 4, In contrast to standard Mixture of Experts (MoE) models—where all experts share the same structure and only differ by routing—Additive AI defines an exponential family of expert models ( ( 2^n ) variants) that differ structurally through binary decisions applied to + (addition) operations. Each + operation in the architecture acts as a gate, controlling whether a submodule (e.g., subactivation, residual block) is included or skipped, resulting in a diverse set of model structures with shared weights.

Finally In Exp 5, we introduce a training booster and model compression approach inspired by the concept of strong and weak neurons. Lightweight models (with reduced dimensionality) are trained in parallel on data clusters for a few epochs, then synchronized to a larger base model rest of epochs larger model will train on pipeline or zero offload training. This approach accelerates early-stage convergence and enables scalable model compression. The compression strategy draws on ideas from our prior work, DiasDNN-VD, which used variational dropouts in a divide-and-conquer fashion for asynchronous DNN training. Additive AI thus offers a unified path for structure-aware training, compression, and performance optimization for large-scale language models.

# MLP
<table>
  
  <tr>
    <td valign="top"><img src="./images/mlp1.jpeg"></td>
    
    
  </tr>
  <td valign="top"><img src="./images/mlp1.jpeg"></td>
  <tr>
    
  </tr>
 </table>

# Attention
<table>
  
  <tr>
    <td valign="top"><img src="./images/selfattention1.jpeg"></td>
    
    
  </tr>
  <td valign="top"><img src="./images/selfattention2.jpeg"></td>
  <tr>
    
  </tr>
 </table>