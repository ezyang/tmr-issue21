% -*- LaTeX -*-
\documentclass{tmr}

\usepackage{mflogo}
\usepackage{fancyvrb}

%include polycode.fmt

\title{A Functional Approach to Neural Networks}
\author{Amy de Buitl\'eir\email{amy.butler@@ericsson.com}}
\author{Michael Russell\email{mrussell@@ait.ie}}
\author{Mark Daly\email{mdaly@@ait.ie}}

\newcommand{\authornote}[3]{{\color{#2} {\sc #1}: #3}}
\newcommand\ezy[1]{\authornote{edward}{blue}{#1}}

\begin{document}

\begin{introduction}
We describe an approach to implementing a neural network \ezy{It's probably worth briefly explaining why neural networks are interesting}
in a functional programming language,
using a basic back-propagation algorithm for illustration.
We highlight the benefits of a purely functional approach for both the development
and testing of neural networks.
Although the examples are coded in Haskell,
the techniques described should be applicable to any functional programming language.
\end{introduction}

\ezy{I've fixed any comma splices that I found. Remember the general
rule for a comma splice is, if there is a comma and then a conjunction,
then the two parts split apart should stand alone as complete sentences.
If they don't, then there should not be a comma.}

\ezy{Also, an aside: you can use the dollar sign to demarcate mathematical equations;
it's a lot less typing than begin math!}

\section{Back-propagation}
\textit{Back-propagation} is a common method of training artificial \ezy{Is this adverb necessary?} neural networks.
After an input pattern is propagated forward through the network to produce an output pattern,
the output pattern is compared to the target (desired) pattern,
and the error is then propagated backward.
During the back-propagation phase, each neuron's contribution to the error is calculated,
and the network configuration can be modified with the goal of reducing future errors.
Back-propagation is a supervised training method, so the correct answers for the training set 
must be known in advance
or be calculable.
In this paper, we use a simple ``no-frills'' back-propagation algorithm;
this is sufficient for demonstrating a functional approach to neural networks.

\section{Neural networks}
\label{sec:neuralNetOverview}

\subsection{An artificial neuron}

Conceptually, \ezy{I think this word is unnecessary} the basic building block of an artificial neural network is the neuron, 
shown in Figure \ref{fig:neuron}.
It is characterized by~\cite{gurney_introduction_1997} \ezy{This cite style is awkward; I wonder what a good convention here is}
\begin{itemize}
        \item a set of inputs \begin{math}x_i\end{math}, usually more than one;
        \item a set of weights \begin{math}w_i\end{math} associated with each input;
        \item the weighted sum of the inputs \begin{math}a = \Sigma x_i w_i\end{math};
        \item an activation function \begin{math}f(a)\end{math} which acts on the weighted sum of the inputs,
 and determines the output;
        \item a single output \begin{math}y = f(a)\end{math}.
\end{itemize}

\begin{figure}[H]
  \centering
  \includegraphics[scale=0.25,keepaspectratio=true]{images/neuron.pdf}
  \vspace{5pt}
  \caption{An artificial neuron.}
  \label{fig:neuron}
\end{figure}

\subsection{A simple network}

The most common type of artificial neural network is a \textit{feed-forward network}.
In a feed-forward network, the neurons are grouped into layers,
as shown in Figure \ref{fig:layers}.
Each neuron feeds its output forward to every neuron in the following layer.
There is no feedback from a later layer to an earlier one
and no connections within a layer, e.g. there are no loops.
The elements of the \textit{input pattern} to be analyzed are presented to a \textit{sensor layer},
which has one neuron for every component of the input.
The sensor layer performs no processing; it merely distributes its input to the next layer.
After the sensor layer comes one or more \textit{hidden layers};
the number of neurons in these layers is arbitrary.
The last layer is the \textit{output layer};
the outputs from these neurons form the elements of the \textit{output pattern}.
Hence, the number of neurons in the output layer must match the desired length of the output pattern.

\begin{figure}[H]
  \centering
  \includegraphics[scale=0.30,keepaspectratio=true]{images/neuralNet.pdf}
  \vspace{5pt}
  \caption{A simple neural network.}
  \label{fig:layers}
\end{figure}

\subsection{Training the network}
The \textit{error} \ezy{Of a what?} is a function of the vector difference \ezy{Looking at the level of detail you're using in the explanation here, it's probably worth defining this} between the output pattern 
and the \textit{target pattern} (desired output).
The network can be trained by adjusting the network weights with the goal of reducing the error.
Back-propagation is one technique for choosing the new weights.~\cite{rumelhart-learning-1986}
This is a \textit{supervised learning} process:
the network is presented with both the input pattern as well as the target pattern.
The error from the output layer is propagated backward through the hidden layers
in order to determine each layer's contribution to the error,
a process is illustrated in Figure~\ref{fig:backprop}.
The weights in each layer are then adjusted to reduce the error for that input pattern.

\begin{figure}[H]
  \centering
  \includegraphics[scale=0.60,keepaspectratio=true]{images/backprop.pdf}
  \vspace{10pt}
  \caption{Back-propagation.}
  \label{fig:backprop}
\end{figure}

\section{Building a neural network}
\label{sec:buildNetwork}

\subsection{Building a neuron}

In this implementation, we use matrices to represent the weights for the neurons in each layer.
The matrix calculations are performed using Alberto Ruiz's \verb+hmatrix+~\cite{ruiz_hmatrix_web, ruiz_simple}, 
a purely functional Haskell interface to basic matrix computations and other
numerical algorithms in GSL~\cite{galassi_gnu_2009}, 
BLAS~\cite{national_science_foundation_blas_web, dongarra_blas_2002} and 
LAPACK~\cite{national_science_foundation_lapack_web, anderson_lapack_1999}. 
With a matrix-based approach, there is no need for a structure to represent a single neuron. Instead, the
implementation of the neuron is distributed among the following entities

\begin{itemize}
        \item the inputs from the previous layer
        \item the output to the next layer
        \item a column in the weight matrix
        \item an activation function 
(in this implementation, the same function is used for all neurons in all layers except the sensor layer)
\end{itemize}

For the weight matrix, we use the \verb+Matrix+ type provided by \verb+hmatrix+.
The inputs, outputs and patterns are all column vectors.
We use the \verb+Matrix+ type for these as well,
but we introduce the type synonym \ezy{I unbolded this, see also later; I don't think that type synonyms or record syntax are essential features of Haskell which made implementing neural networks easier, while first-class functions do seem to deserve the emphasis.} \verb+ColumnVector+.
In Haskell, the \verb+type+ keyword defines an alternative name for an existing type;
it does not define a new type. (A complete code listing is available from the authors, along with a sample character recognition application.) \ezy{A link to this implementation would be good!}
\begin{code}
type ColumnVector a = Matrix a
\end{code}

The final element we need to represent the neuron \ezy{Wording here is awkward} is the activation function.
Here, we encounter one of the advantages of a functional approach.
Like most most functional programming languages,
Haskell supports \emph{first-class functions}; 
a function can be used in the same way as any other type of value.
It can be passed as an argument to another function, stored in a data structure,
or returned as result of function evaluation.
Hence, we don't need to do anything special to allow this neural network to use
any activation function chosen by the user.
The activation function can be supplied as an argument at the time the network is created.

It is convenient to create a structure to hold both the activation function
and its first derivative.
(The back-propagation algorithm requires that the activation function be differentiable,
and we will need the derivative to apply the back-propagation method.)
This helps to reduce the chance that the user will change the activation function 
and forget to change the derivative. \ezy{Rhetorical question: why can't the derivative be computed automatically?}
We define this type using Haskell's record syntax \ezy{Same as above unbolding},
and include a string to describe the activation function being used.

\begin{code}
data ActivationSpec 
    = ActivationSpec
        {
            asF :: Double -> Double
          , asF' :: Double -> Double
          , desc :: String
        }
\end{code}

The first field, \verb+asF+, is the activation function, which takes a \verb+Double+
(double precision, real floating-point value) 
as input and returns a \verb+Double+.
The second field, \verb+asF'+, is the first derivative.
It also takes a \verb+Double+ and returns a \verb+Double+.
The last field, \verb+desc+, is a \verb+String+ value containing a description of the function.

Accessing the fields of a value of type \verb+ActivationSpec+ is straightforward.
For example, if the name of the value \ezy{Word choice} is \verb+s+,
then its activation function is \verb+asF s+,
its first derivative is \verb+asF' s+,
and its description is \verb+desc s+.

As an example of how to create a value of the type \verb+ActivationSpec+,
here is one for the identity function \begin{math}f(x) = x\end{math}, 
whose first derivative is \begin{math}f'(x) = 1\end{math}.

\begin{code}
identityAS = ActivationSpec
    {
        asF = id
      , asF' = const 1
      , desc = "identity"
    }
\end{code}

The function \verb+id+ is Haskell's predefined identity function.
The definition of \verb+asF'+ may seem puzzling.
The first derivative of the identity function is 1, but we cannot
simply write \verb+asF' = 1+. Why not?
Recall that the type signature of \verb+asF'+ is \verb+Double -> Double+,
so we need to assign an expression to it that takes a \verb+Double+ and returns a \verb+Double+.
However, \verb+1+ is just a single number.
It could be of type \verb+Double+, but not \verb+Double -> Double+.
To solve this issue, we make use of 
the predefined \verb+const+ function, which takes two parameters
and returns the first, ignoring the second.
By partially applying it (supplying \verb+1+ as the first parameter),
we get a function that takes a single parameter and always returns the value \verb+1+.
So the expression \verb+const 1+ can satisfy the type signature \verb+Double -> Double+.

The hyperbolic tangent is a commonly-used activation function;
the appropriate \verb+ActivationSpec+ is defined below.
 
\begin{code}
tanhAS :: ActivationSpec
tanhAS = ActivationSpec
    {
        asF = tanh
      , asF' = tanh'
      , desc = "tanh"
    }

tanh' x = 1 - (tanh x)^2
\end{code} % Note to EZY: figure out why caret is being rendered funny

At this point, we taken advantage of Haskell's support for \emph{first-class functions} to
store functions in a record structure and to pass functions as parameters to
another function (in this case, the \verb+ActivationSpec+ constructor).

\subsection{Building a neuron layer}

To define a layer in the neural network, we use a record structure
containing the weights and the activation specification.
The weights are stored in an \begin{math}n \times m\end{math} matrix,
where \begin{math}n\end{math} is the number of inputs and \begin{math}m\end{math} is the number of neurons.
The number of outputs from the layer is equal to the number of neurons, \begin{math}m\end{math}.

\begin{code}
data Layer
    = Layer
        {
          lW :: Matrix Double
        , lAS :: ActivationSpec
    }
\end{code}

The weight matrix, \verb+lW+, has type \verb+Matrix Double+.
This is a matrix whose element values are double-precision floats.
This type \ezy{While technically correct, it's a kind of awkward phrasing, because it kind of implies only the type is defined in hmatrix, which is not true; the implementation is there too!} is defined in the \verb+hmatrix+ package.
The activation specification, \verb+lAS+ uses the type \verb+ActivationSpec+, defined earlier.
Again we use the support for first-class functions \ezy{after the first bolding I'd hazard that we don't need to bold it anymore};
to create a value of type \verb+Layer+, 
we pass a record containing function values into another function,
the \verb+Layer+ constructor.

\subsection{Assembling the network}

The network consists of a list of layers and a parameter to control the rate at which the network 
learns new patterns.

\begin{code}
data BackpropNet
  = BackpropNet
    {
      layers :: [Layer]
    , learningRate :: Double
    }
\end{code}

The notation \verb+[Layer]+ indicates a list whose elements are of type \verb+Layer+.
We can create a value of type \verb+BackpropNet+ using an expression such as the following.

\begin{code}
layer = Layer { lW=w, lAS=s }
\end{code}
\ezy{Since you've already used record assignment syntax, either this explanation is redundant or needs to be moved up earlier.}

Of course, the number of outputs from one layer must match the number of inputs to the next layer.
We ensure this by not exporting \ezy{word choice (negation is awkward)} the \verb+BackpropNet+ constructor outside the module \ezy{also awkward; when you export, it's always going outside the module}
and instead requiring the user to call a special function to construct the network. \ezy{The usual term for this is a smart constructor}
First, we address the problem of how to verify that the dimensions of 
a consecutive pair of network layers is compatible.
The following function will report an error if a mismatch is detected.

\begin{code}
checkDimensions :: Matrix Double -> Matrix Double -> Matrix Double
checkDimensions w1 w2 =
  if rows w1 == cols w2
       then w2
       else error "Inconsistent dimensions in weight matrix"
\end{code}

Assuming that no errors are found, \verb+checkDimensions+ simply returns the second layer in a pair. \ezy{The reason why the second layer should be returned is not obvious until we learn about scans a few sentences down.}

The constructor function should invoke \verb+checkDimensions+ on each pair of layers.
In an imperative language, a for loop would typically be used.
In functional languages, a recursive function could be used to achieve the same effect.
However, there is a more straightforward solution using an operation called a \textit{scan}.
There are several variations on this operation, 
and it can proceed either from left to right, or from right to left.
We've chosen the predefined operation \verb+scanl1+, read ``scan-ell-one'' (not ``scan-eleven'').
\ezy{I probably would have placed the code snippet here}
The l indicates that the scan starts from the left, 
and the 1 indicates that we want the variant that takes no starting value.
\begin{code}
scanl1 f [x1, x2, x3, ...] == [x1, f x1 x2, f (f x1 x2) x3, ...]
\end{code}

Applying \verb+scanl1 checkDimensions+ to a list of weight matrices gives the following result
(again assuming no errors are found).

\begin{code}
scanl1 checkDimensions [w1, w2, w3, ...] 
  == [w1, checkDimensions w1 w2, 
         checkDimensions (checkDimensions w1 w2) w3, ...]
  == [w1, w2, w3, ...]
\end{code}

Therefore, if the dimensions of the weight matrices are consistent, this operation
simply returns the list of matrices, e.g. it is the identity function.
The next task is to create a layer for each weight matrix supplied by the user.
In an imperative language, we might operate on each element in the weight matrix list using a for loop. \ezy{A bit repetitive, perhaps.}
Instead, we will use the \verb+map+ function.
The expression \verb+map buildLayer checkedWeights+ will return a new list,
where each element is the result of applying the function \verb+buildLayer+ 
to the corresponding element in the list of weight matrices.
The definition of \verb+buildLayer+ is simple, it merely invokes the constructor for the type
\verb+Layer+, defined earlier.

\begin{code}
buildLayer w = Layer { lW=w, lAS=s }
\end{code}

Using the operations discussed above, we can now define the constructor function, \verb+buildBackpropNet+.

\begin{code}
buildBackpropNet 
  :: Double 
  -> [Matrix Double] 
  -> ActivationSpec 
  -> BackpropNet
buildBackpropNet lr ws s = BackpropNet { layers=ls, learningRate=lr }
  where checkedWeights = scanl1 checkDimensions ws
        ls = map buildLayer checkedWeights
        buildLayer w = Layer { lW=w, lAS=s }
\end{code}

\ezy{This code needs some seqs: the entire list should be bottom on a failed
checkDimension, not just the failed layer}

The primary advantage of using functions such as \verb+map+ and \verb+scanl1+ is not
that they save a few lines of code over an equivalent \textit{for loop}, but that these functions
more clearly indicate the programmer's intent. \ezy{Rhetorical question: Why?}

\section{Running the Network}
\label{sec:propagation}

\subsection{A closer look at the network structure}

The network \ezy{What network are you referring to?} consists of multiple layers of neurons, numbered from \begin{math}0\end{math} 
to \begin{math}L\end{math}, as illustrated in Figure \ref{fig:propagation}.
Each layer is fully connected to the next layer.
Layer \begin{math}0\end{math} is the sensor layer.
(It performs no processing; 
each neuron receives one component of the input vector \begin{math}\mathbf{x}\end{math} and 
distributes it, unchanged, to the neurons in the next layer.)
Layer \begin{math}L\end{math} is the output layer.
The layers \begin{math}l = 1..(L-1)\end{math} are hidden layers.
\begin{math}z_{lk}\end{math} is the output from neuron \begin{math}l\end{math} in layer \begin{math}l\end{math}.

\begin{figure}[H]
  \centering
  \includegraphics[scale=0.40,keepaspectratio=true]{images/backpropNet.pdf}
  \vspace{5pt}
  \caption{Propagation through the network.}
  \label{fig:propagation}
\end{figure}

We use the following notation:

\begin{itemize}
  \item \begin{math}x_i\end{math} is the \begin{math}i\end{math}th component of the input pattern;
  \item \begin{math}z_{li}\end{math} is the output of the \begin{math}i\end{math}th neuron in layer \begin{math}l\end{math};
  \item \begin{math}y_i\end{math} is the \begin{math}i\end{math}th component of the output pattern.
\end{itemize}


\subsection{Propagating through one layer}

The activation \ezy{Term not defined} of neuron \begin{math}k\end{math} in layer \begin{math}l\end{math} is

\ezy{Is the special case here necessary? In some treatments of neural networks I've seen, appropriate settings of $w$ are sufficient to make the second equation work out.  Obviously, if you do that you'll have to change your code too, but with any luck that would be for the better}

\begin{displaymath}
a_{0k} = x_k
\end{displaymath}

\begin{displaymath}
a_{lk} = \sum_{j=1}^{N_{l-1}} w_{lkj}z_{l-1,j}~~~~~~~~~~l > 0
\end{displaymath}

where 
\begin{itemize}
\item\begin{math}N_{l-1}\end{math} is the number of neurons in layer \begin{math}l-1\end{math}.
\item\begin{math}w_{lkj}\end{math} is the weight applied by the neuron \begin{math}k\end{math} in layer \begin{math}l\end{math} to the input received from neuron \begin{math}j\end{math} in layer \begin{math}l-1\end{math}.
(Recall that the sensor layer, layer \begin{math}0\end{math}, simply passes along its inputs without change.)
\end{itemize}

We can express the activation for layer \begin{math}l\end{math} using a matrix equation.

\begin{displaymath}
\mathbf{a_l} = \left\{ 
\begin{array}{l l}
  \mathbf{x} & \quad l=0\\
\\
  \mathbf{W}_l \mathbf{x} & \quad l>0\\
\end{array} \right.
\end{displaymath}

The output from the neuron is

\begin{displaymath}
z_{lk} = f(a_{lk})
\end{displaymath}

where \begin{math}f(a)\end{math} is the activation function.
For convenience, we define the function \verb+mapMatrix+ which applies a function to each element of a matrix (or column vector).
This is analogous to Haskell's \verb+map+ function. \ezy{In fact, matrices are functors!}
(The definition of this function is in the appendix.)
Then we can calculate the layer's output using the Haskell expression \verb+mapMatrix f a+, where \verb+f+ is the activation function.

If we've only propagated the input through the network, all we need is the output from the final layer, 
\begin{math}\mathbf{z}_L\end{math}.
However, we will keep the intermediate calculations
because they will be required during the back-propagation pass.
We will keep all of the necessary information in the following record structure.
Note that anything between the symbol \verb+--+ and the end of a line is a comment
and is ignored by the compiler.

\begin{code}
data PropagatedLayer
    = PropagatedLayer
        {
          -- The input to this layer
          pIn :: ColumnVector Double
          -- The output from this layer
        , pOut :: ColumnVector Double
          -- First derivative of the activation function for this layer
        , pF'a :: ColumnVector Double
          -- The weights for this layer
        , pW :: Matrix Double
          -- The activation specification for this layer
        , pAS :: ActivationSpec
        }
    | PropagatedSensorLayer
        {
          -- The output from this layer
          pOut :: ColumnVector Double
        }
\end{code}

This structure has two variants.
For the sensor layer 
(\verb+PropagatedSensorLayer+), 
the only information we need is the output,
which is identical to the input.
For all other layers (\verb+PropagatedLayer+), 
we need the full set of values.
Now we are ready to define a function to propagate through a single layer.

\begin{code}
propagate
  :: PropagatedLayer
  -> Layer
  -> PropagatedLayer
propagate layerJ layerK = PropagatedLayer
        {
          pIn = x
        , pOut = y
        , pF'a = f'a
        , pW = w
        , pAS = lAS layerK
        }
  where x = pOut layerJ
        w = lW layerK
        a = w <> x
        f = asF ( lAS layerK )
        y = mapMatrix f a
        f' = asF' ( lAS layerK )
        f'a = mapMatrix f' a
\end{code}

The operator \verb+<>+ performs matrix multiplication; it is defined in the \verb+hmatrix+ package.


\subsection{Propagating through the network}

To propagate through the entire network, \ezy{Awkward: what are we propagating?}
we create a sensor layer to provide the inputs
and use another \textit{scan} operation,
this time with \verb+propagate+.
The \verb+scanl+ function is similar to the \verb+scanl1+ function,
except that it takes a starting value.

\begin{code}
scanl f z [x1, x2, ...] == [z, f z x1, f (f z x1) x2), ...] 
\end{code}

In this case, the starting value is the sensor layer.

\begin{code}
propagateNet
  :: ColumnVector Double
  -> BackpropNet
  -> [PropagatedLayer]
propagateNet input net = tail calcs
  where calcs = scanl propagate layer0 (layers net)
        layer0 = PropagatedSensorLayer{ pOut=validatedInputs }
        validatedInputs = validateInput net input
\end{code}

The function \verb+validateInput+ verifies that the input vector has the correct length
and that the elements are within the range [0,1].
Its definition is straightforward.

\section{Training the network}
\label{sec:backprop}

\subsection{The back-propagation algorithm}

We use the matrix equations for basic back-propagation as formulated by Hristev
~\cite[Chapter 2]{hristev_ann_1998}. \ezy{It's probably worth noting here how much you are going to recap, and how much you are going to defer to this reference.  I found the exposition here not very clear (and it certainly would not be clear to someone who has never used neural networks before), but if the point is to get the equations down so we can transcribe them into Haskell, that might be OK.}
The back-propagation algorithm requires that we operate on each layer in turn, \ezy{This seems not quite accurate, since you first go forwards then you go backwards}
using the results of the operation on one layer as input to the operation on the next layer.
The input vector \begin{math}\mathbf{x}\end{math} is propagated \emph{forward} through the network,
resulting in the output vector \begin{math}\mathbf{z_L}\end{math},
which is then compared to the target vector \begin{math}\mathbf{t}\end{math} (the desired output).
The resulting error, \begin{math}\mathbf{z_L} - \mathbf{t}\end{math} is then propagated
\emph{backward} to determine the corrections to the weight matrices:

\begin{equation}
\label{eq:weightUpdate}
W_{new}= W_{old}-\mu\nabla E
\end{equation}

where \begin{math}\mu\end{math} is the learning rate, and \begin{math}E\end{math} is the error function.
For \begin{math}E\end{math}, we can use
the sum-of-squares error function, defined below.

\begin{displaymath}
E(W) \equiv \frac{1}{2} \sum_{q=1}^{N_L} [z_{Lq}(x) - t_q(x)]^2
\end{displaymath}

where \begin{math}z_{Lq}\end{math} is the output from neuron q in the output layer (layer L).
The error gradient for the last layer is given by:

\begin{equation}
\label{eq:dazzleL}
\nabla_{z_L}E = \mathbf{z}_{L}(x) - \mathbf{t}
\end{equation}
The error gradient for a hidden layer can be calculated recursively according to 
the equations below.
(See~\cite[Chapter 2]{hristev_ann_1998} for the derivation.)

\begin{displaymath}
(\nabla E)_l = [\nabla_{z_l}E \odot f'(a_l)] \cdot z^T_{l-1}
~~~~~ \textit{for layers }
l=\overline{1,L}
\end{displaymath}

\ezy{I think the z above should be bold, since it's a vector.  Also, the paren-subscribe delta notation is a little odd.}

\begin{equation}
\label{eq:dazzle}
\nabla_{z_l}E = W^t_{l+1} \cdot [\nabla_{z_{l+1}}E \odot f'(a_{l+1})]
~~~~ \textit{calculated recursively from L-1 to 1}
\end{equation}

The symbol \begin{math}\odot\end{math} is the \textit{Hadamard}, or element-wise product.


\subsection{Back-propagating through a single layer}

The result of back-propagation through a single layer is stored in the structure below.
The expression \begin{math}\nabla_{z_l}E\end{math} is not easily represented in ASCII text,
so the name ``dazzle'' is used in the code.

\begin{code}
data BackpropagatedLayer
    = BackpropagatedLayer
        {
          -- Del-sub-z-sub-l of E
          bpDazzle :: ColumnVector Double
          -- The error due to this layer
        , bpErrGrad :: Matrix Double
          -- The value of the first derivative of the activation function function for this layer
        , bpF'a :: ColumnVector Double
          -- The input to this layer
        , bpIn :: ColumnVector Double
          -- The output from this layer
        , bpOut :: ColumnVector Double
          -- The weights for this layer
        , bpW :: Matrix Double
          -- The activation specification for this layer
        , bpAS :: ActivationSpec
        }
\end{code}

The next step is to define the \verb+backpropagate+ function.
For hidden layers, we use Equation (\ref{eq:dazzle}).

\begin{code}
backpropagate
  :: PropagatedLayer
  -> BackpropagatedLayer
  -> BackpropagatedLayer
backpropagate layerJ layerK = BackpropagatedLayer
        {
          bpDazzle = dazzleJ            
        , bpErrGrad = errorGrad dazzleJ f'aJ inputJ
        , bpF'a = pF'a layerJ
        , bpIn = pIn layerJ
        , bpOut = pOut layerJ
        , bpW = pW layerJ
        , bpAS = pAS layerJ
        }
    where dazzleJ = wkT <> (dazzleK * f'aK)
          dazzleK = bpDazzle layerK
          wkT = trans ( bpW layerK )
          errK = bpErrGrad layerK
          f'aK = bpF'a layerK
          f'aJ = pF'a layerJ
          inputJ = pIn layerJ
errorGrad :: ColumnVector Double -> ColumnVector Double -> 
  ColumnVector Double -> Matrix Double
errorGrad dazzle f'a input = (dazzle * f'a) <> trans input
\end{code}
\ezy{The bits where the record is simply inheriting the old values from \verb|layerJ|; at the very least you could reduce line noise by using record update syntax instead of record creation, but also consider if there is an easy way to change the ADT so that these assignments are not necessary.  Also, if you're trying to make the point that Haskell looks a lot like the math which it is computing, it's probably worth recapping the relevant equations near the code so a direct visual comparison can be made (also, avoid temporary variables)}

The function \verb+trans+ calculates the transpose of a matrix.
The operator \verb+*+ used in the definition of \verb+dazzleJ+ appears between
two column vectors, \verb+dazzleK+ and \verb+f'aK+,
so it calculates the Hadamard (element-wise) product rather than a scalar product.
The final layer uses Equation (\ref{eq:dazzleL}).

\begin{code}
backpropagateFinalLayer
    :: PropagatedLayer
    -> ColumnVector Double
    -> BackpropagatedLayer
backpropagateFinalLayer l t = BackpropagatedLayer
        {
          bpDazzle = dazzle            
        , bpErrGrad = errorGrad dazzle f'a input 
        , bpF'a = pF'a l
        , bpIn = pIn l
        , bpOut = pOut l
        , bpW = pW l
        , bpAS = pAS l
        }
        where dazzle =  pOut l - t
              f'a = pF'a l
              input = pIn l 
\end{code}


\subsection{Back-propagating through the network}

We have already introduced the \verb+scanl+ function, which operates on an array from left to right.
For the back-propagation pass, we will use \verb+scanr+, which operates from right to left.
Figure \ref{fig:implementation} illustrates how \verb+scanl+ and \verb+scanr+ will act on the neural network.
The boxes labeled \verb+pc+ and \verb+bpc+ represent the result of each propagation operation
and back-propagation operation, respectively.
Viewed in this way, it is clear that \verb+scanl+ and \verb+scanr+
provide a layer of abstraction that is ideally suited to back-propagation.

\begin{figure}[H]
  \centering
  \includegraphics[scale=0.70,keepaspectratio=true]{images/scan.pdf}
  \vspace{10pt}
  \caption{A schematic diagram of the implementation.}
  \label{fig:implementation}
\end{figure}

The definition of the \verb+backpropagateNet+ function is very similar to that of \verb+propagateNet+.
\begin{code}
backpropagateNet
  :: ColumnVector Double
  -> [PropagatedLayer]
  -> [BackpropagatedLayer]
backpropagateNet target layers =
    scanr backpropagate layerL hiddenLayers
  where hiddenLayers = init layers
        layerL = backpropagateFinalLayer (last layers) target
\end{code}
\ezy{Ick, not a fan of init and last}

\subsection{Updating the weights}
\label{sec:updateWeights}

After the back-propagation calculations have been performed,
the weights can be updated using Equation (\ref{eq:weightUpdate}).

\begin{code}
update :: Double -> BackpropagatedLayer -> Layer
update rate layer = Layer
        {
          lW = wNew 
        , lAS = bpAS layer
        }
    where wOld = bpW layer
          delW = rate .* bpErrGrad layer
          wNew = wOld - delW
\end{code}

The operator \verb+.*+ performs element-wise multiplication of a matrix by a scalar.

\section{A functional approach to testing}
\label{sec:testing}

In traditional unit testing, the tester writes code to test \ezy{tester--test; awkward} individual cases.
For some applications, determining the desired result for each test case can be time-consuming,
which limits the number of cases that will be tested.

Property-based testing tools such as QuickCheck~\cite{claessen_quickcheck_2000} take a different approach.
The tester defines properties that should hold for all cases,
or, at least, for all cases satisfying certain criteria.
In most cases, QuickCheck can automatically generate suitable pseudo-random test data
and verify that the properties are satisfied, saving the tester's time.
In keeping with the test-driven development~\cite{beck-test-driven-2003} methodology,
these properties could be defined and the tests automated before
any code is written for the unit under test. \ezy{A little bit of a non-sequitur}

QuickCheck can also be invaluable in isolating faults,
and finding the simplest possible test case that fails.
This is partially due to the way QuickCheck works: it begins with ``simple'' cases
(for example, setting numeric values to zero or using zero-length strings and arrays),
and progresses to more complex cases.
When a fault is found, it is typically a minimal failing case.
If the default functions provided by QuickCheck for generating pseudo-random test data are not suitable,
the tester can write custom functions. \ezy{Note that modern QuickCheck also supports shrinking, which methodically reduces test-cases in an attempt to find a minimal test-case}

An in-depth look at QuickCheck is beyond the scope of this article. 
Instead, we will show one example to illustrate the value of property-based testing.
What properties should a neural network satisfy, no matter what input data is provided?
One property it should satisfy is that \ezy{Awkward} if the network is trained once 
with a given input pattern and target pattern
and immediately run on the same input pattern, the error should be reduced.
Another way of saying this is that \ezy{Also awkward} training should reduce the error in the output layer, 
unless the error is negligible to begin with.
We define the property as follows.

\ezy{You pull a switcheroo here: the desirable property described above is for an entire neural network, but here, only a single layer is being tested. The further down-side of oding this is you need a lot of support code in where to actually setup the test.  (Admittedly, some of this would be helped if you defined some helper functions...)}

\ezy{The function definition here is a little non-standard; actually, you can write the function in the normal curried form; QuickCheck has appropriate typeclass magic to make it all work out. (This also means the \verb|forAll| is not necessary.}

\ezy{Better style when making pipelines of composed functions is to use dots except for the last one, which should be a dollar.  However, if you're writing for not necessarily a Haskell audience (which some parts of the article seemed to imply), then maybe this syntax should be avoided, or at least explained.}

\begin{code}
-- Property: training reduces error in the final (output) layer
trainingReducesFinalLayerError :: 
  (ColumnVector Double, Layer, ColumnVector Double) -> Property
trainingReducesFinalLayerError (x, l, t) = 
    classifyRange "len x " n 0 25 $ 
    classifyRange "len x " n 26 50 $ 
    classifyRange "len x " n 51 75 $ 
    classifyRange "len x " n 76 100
    (errorAfter < errorBefore || errorAfter < 0.01)
        where n = inputWidth l
              pl0 = PropagatedSensorLayer{ pOut=x }
              pl = propagate pl0 l
              bpl = backpropagateFinalLayer pl t
              errorBefore = P.magnitude (t - pOut pl)
              lNew = update 0.0000000001 bpl -- make sure we don't overshoot the mark
              plNew = propagate pl0 lNew
              errorAfter =  P.magnitude (t - pOut plNew)

-- Testable property:
-- Training reduces error in the final (output) layer
prop_trainingReducesFinalLayerError :: Property
prop_trainingReducesFinalLayerError =
    forAll arbLayerTestData trainingReducesFinalLayerError
\end{code}

\ezy{\verb|arbLayerTestData| may be worth including in the article itself}

The function \verb+trainingReducesFinalLayerError+ takes a tuple consisting of an input pattern \verb+x+,
an output layer \verb+l+, and a target pattern \verb+t+, and returns a boolean to indicate
whether or not the property holds.
This particular property only checks that training works for an output layer;
our complete implementation tests other properties, including the effect of training on hidden layers.
The resulting property returns true if the error before training is less than the error after,
or if the error is already negligible (less than 0.001 for this test). \ezy{Repetitive.}
The \verb+classifyRange+ statements are useful when running the tests interactively;
they display a brief report indicating the distribution of the test inputs.
The function \verb+trainingReducesFinalLayerError+ specifies that 
a custom generator for pseudo-random test data, \verb+arbLayerTestData+,
is to be used.
The generator \verb+arbLayerTestData+ ensures that the ``simple'' test cases that QuickCheck starts with
consist of short patterns and a network with a small total number of neurons.
We do this so that if there are errors, the first failing test case found
will be easier to analyze. \ezy{Repetitive.}

We can run the test in \verb+GHCi+, an interactive Haskell REPL.

\begin{Verbatim}
ghci> quickCheck prop_trainingReducesFinalLayerError
+++ OK, passed 100 tests:
62% len x 0..25
24% len x 26..50
12% len x 51..75
 2% len x 76..100
\end{Verbatim}

By default, QuickCheck runs 100 test cases.
Of these, 62\% of the patterns tested were of length 25 or less.
We can request more test cases:
the test of 10,000 cases below ran in 20 seconds on a 3.00GHz quad core processor running Linux.
It would not have been practical to write unit tests for this many cases,
so the benefit of property-based testing as a supplement to unit testing is clear.

\begin{Verbatim}
ghci> quickCheckWith Args{replay=Nothing, maxSuccess=10000, 
maxDiscard=100, maxSize=100} prop_trainingReducesFinalLayerError
+++ OK, passed 10000 tests:
58% len x 0..25
25% len x 26..50
12% len x 51..75
 3% len x 76..100
\end{Verbatim}

\section{Conclusions}
\label{sec:conclusions}

We have seen that Haskell provides operations
such as \verb+map+, \verb+scanl+, \verb+scanr+, and their variants, 
that are particularly well-suited for implementing neural networks
and back-propagation.
These operations are not unique to Haskell;
they are part of a category of functions commonly provided by functional programming languages
to factor out common patterns of recursion and perform the types of operations that would
typically be performed by loops in imperative languages.
Other operations in this category include \textit{folds},
which operate on lists of values using a combining function to produce a single value, and \textit{unfolds},
which take a starting value and a generating function, and produce a list.

Functional programming has some clear advantages for implementing mathematical solutions.
There is a straightforward relationship between the mathematical equations and the 
corresponding function definitions. \ezy{See my earlier comment about keeping the equations close to the code}
Note that in the back-propagation example, we merely created data structures
and wrote definitions for the values we needed.
At no point did we provide instructions on how to sequence the operations.
The final results were defined in terms of intermediate results,
which were defined in terms of other intermediate results,
eventually leading to definitions in terms of the inputs.
The compiler is responsible for either finding an appropriate sequence in which to apply 
the definitions or reporting an error if the definitions are incomplete.

Property-based testing has obvious benefits.
With minimal effort, we were able to test the application very thoroughly.
But the greatest advantage of property-based testing may be its ability to isolate
bugs and produce a minimal failing test case.
It is much easier to investigate a problem when the matrices involved in calculations are small.

Functional programming requires a different mind-set than imperative programming,
which can lead to a conceptual mismatch. \ezy{Awkward}
Textbooks on neural network programming usually provide derivations and definitions,
but with the ultimate goal of providing an algorithm for each technique discussed.
The functional programmer needs only the definitions,
but it would be wise to read the algorithm carefully in case it contains additional information
not mentioned earlier.

Functional programming may not be suited to everyone, or to every problem.
However, some of the concepts we have demonstrated can be applied in imperative languages.
Some imperative languages have borrowed features such as first-class functions, maps, scans and folds 
from functional languages.
And some primarily functional languages provide mechanisms for doing object-oriented programming. \ezy{It's worth noting that one such language is OCaml, but many reputable OCaml shops explicitly ban use of the object-oriented mechanisms.}
Crossover between these two paradigms is beneficial because it provides programmers with more ways to approach problems. \ezy{Is this established by the article?}

\ezy{I'll double-comment: a link to the code would be great here!}

\bibliography{deBuitleir}

\end{document}
