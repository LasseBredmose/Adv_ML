**Meeting 1/3**
 * Resize the picture - Strecthing the picture
 * Dataloader should be done for next time
 * Difficult task to classify normal and abnormal bones
 * Start by classifying what type of bone we are looking at
 * Easier to explain/classify given our limited bagground in human anatomy
 * To parts after the CNN
 * Do the baysian and do the explanation - Split it up
 * Explanation: 
   * Looking at the gradient through out the network
 * Baysian part:
   * Good certainty on the predictions
   * Laplace approx library: Laplace redux
     * take pytorch library and give approxmiations
 * Bayesian networks are different from bayesian neural netwokrs(we are doing the neural part)
 * The Bayesian network builds upon the CNN - we just keep track on different things and then do the some prediction in the end
 * We don't need to worry so much about the paper - More of a guideline

**Meeting 8/3**
  * Next week get a decent CNN
  * Accuracy of 80% or above
  * 2 layers are probably to litte
  * Laplace redux - paper and documentation [GITHUB](https://arxiv.org/abs/2106.14806)
  * Option 1: 
    * Fully trained network - load the model - tell the package to do it on that
  * Option 2:
    * Do the laplace live - but can be more complex and computationally hard
  * Test the two different ways
  * Laplace on the last fully connected layer


  **Meeting 22/3**
    * 4 pages limit(Martin thinks) - Check with course responsible
      * Describe the methods in a way that someone else from the class can understand the big picture
      * Similair structure as a scientific paper - but pick and choose(4 pages)
      * Don't describe CNN etc, but more the Laplace part and the explanations part
    * Debugging laplace
      * need model.eval()
      * try testint without softmax at last
      * Try and run some test models(from documentation) to test if it even work on the computer
    * Would be cool if we did gradient explanation, other class does class explanation