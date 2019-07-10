# A deep learning algorithm for solving partial differential equations
# Universal approximation theorem
In the mathematical theory of artificial neural networks, the universal approximation theorem states that a feed-forward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of Rn, under mild assumptions on the activation function. The theorem thus states that simple neural networks can represent a wide variety of interesting functions when given appropriate parameters; however, it does not touch upon the algorithmic learnability of those parameters.
One of the first versions of the theorem was proved by George Cybenko in 1989 for sigmoid activation functions
The universal approximation theorem can be expressed mathematically:[2][3][6][7]

Let {\displaystyle \varphi :\mathbb {R} \to \mathbb {R} } {\displaystyle \varphi :\mathbb {R} \to \mathbb {R} } be a nonconstant, bounded, and continuous function. Let {\displaystyle I_{m}} I_m denote the m-dimensional unit hypercube {\displaystyle [0,1]^{m}} [0,1]^{m}. The space of real-valued continuous functions on {\displaystyle I_{m}} I_m is denoted by {\displaystyle C(I_{m})} C(I_{m}). Then, given any {\displaystyle \varepsilon >0} \varepsilon >0 and any function {\displaystyle f\in C(I_{m})} f\in C(I_{m}), there exist an integer {\displaystyle N} N, real constants {\displaystyle v_{i},b_{i}\in \mathbb {R} } v_{i},b_{i}\in {\mathbb  {R}} and real vectors {\displaystyle w_{i}\in \mathbb {R} ^{m}} {\displaystyle w_{i}\in \mathbb {R} ^{m}} for {\displaystyle i=1,\ldots ,N} i=1,\ldots ,N, such that we may define:

{\displaystyle F(x)=\sum _{i=1}^{N}v_{i}\varphi \left(w_{i}^{T}x+b_{i}\right)} F(x)=\sum _{{i=1}}^{{N}}v_{i}\varphi \left(w_{i}^{T}x+b_{i}\right)
as an approximate realization of the function {\displaystyle f} f; that is,

{\displaystyle |F(x)-f(x)|<\varepsilon } 
  | F( x ) - f ( x ) | < \varepsilon
for all {\displaystyle x\in I_{m}} x\in I_{m}. In other words, functions of the form {\displaystyle F(x)} F(x) are dense in {\displaystyle C(I_{m})} C(I_{m}).

This still holds when replacing {\displaystyle I_{m}} I_m with any compact subset of {\displaystyle \mathbb {R} ^{m}} \mathbb {R} ^{m}.


The universal approximation theorem for width-bounded networks can be expressed mathematically as follows:[4]

For any Lebesgue-integrable function {\displaystyle f:\mathbb {R} ^{n}\rightarrow \mathbb {R} } {\displaystyle f:\mathbb {R} ^{n}\rightarrow \mathbb {R} } and any {\displaystyle \epsilon >0} \epsilon >0, there exists a fully-connected ReLU network {\displaystyle {\mathcal {A}}} {\displaystyle {\mathcal {A}}} with width {\displaystyle d_{m}\leq {n+4}} {\displaystyle d_{m}\leq {n+4}}, such that the function {\displaystyle F_{\mathcal {A}}} {\displaystyle F_{\mathcal {A}}} represented by this network satisfies

{\displaystyle \int _{\mathbb {R} ^{n}}\left|f(x)-F_{\mathcal {A}}(x)\right|\mathrm {d} x<\epsilon } {\displaystyle \int _{\mathbb {R} ^{n}}\left|f(x)-F_{\mathcal {A}}(x)\right|\mathrm {d} x<\epsilon }
The theorem of limited expressive power for width- {\displaystyle n} n networks can be expressed mathematically as follows:[4]

For any Lebesgue-integrable function {\displaystyle f:\mathbb {R} ^{n}\rightarrow \mathbb {R} } {\displaystyle f:\mathbb {R} ^{n}\rightarrow \mathbb {R} } satisfying that {\displaystyle \{x:f(x)\neq 0\}} {\displaystyle \{x:f(x)\neq 0\}} is a positive measure set in Lebesgue measure, and any function {\displaystyle F_{\mathcal {A}}} {\displaystyle F_{\mathcal {A}}} represented by a fully-connected ReLU network {\displaystyle {\mathcal {A}}} {\displaystyle {\mathcal {A}}} with width {\displaystyle d_{m}\leq n} {\displaystyle d_{m}\leq n}, the following equation holds:

{\displaystyle \int _{\mathbb {R} ^{n}}\left|f(x)-F_{\mathcal {A}}(x)\right|\mathrm {d} x=+\infty \,or\int _{\mathbb {R} ^{n}}|f(x)|\mathrm {d} x} {\displaystyle \int _{\mathbb {R} ^{n}}\left|f(x)-F_{\mathcal {A}}(x)\right|\mathrm {d} x=+\infty \,or\int _{\mathbb {R} ^{n}}|f(x)|\mathrm {d} x}

# Approximation by Superpositions of a Sigmoidal Function
#  Universal Function Approximation by Deep Neural Nets with Bounded Width and ReLU Activations
#  The expressive power of neural networks
#  Approximation capabilities of multilayer feedforward networks
# Deep Learning
# machine-learning
# Genetic Algorithm
