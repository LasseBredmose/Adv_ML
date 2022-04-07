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
    
**Meeting 29/3**
* Make a table with computation time for different hessian - argue why we use the one we use
* Do we want to do gradient explanation or class activation mapping? Sounds like Mikkel thinks class activation
* Class activations: Which features are important from each class.
* We expect that our CNN will be close to the average of the of our compution explanations as the ECE is quite low already
* We get the explatnations from several sampling instances in BNN
* In practice we get the sampling from the multivariate normal distribution from the BNN?? Where we can extract the means and covariance

**Meeting 7/4**
* Do rotations and (horizontal)flips of the inputs, might help if the small 'annotations' are not all structured the same way. 
  * See if there are other transformations one could do
  * If one might could 'blur' the specific notation regions
* 4-5 conv layers - train again see if it helps
* Don't have a scale or metric to compute the 'accuracy' of the explanations
* The results is really cool. we just need to fix it from here
* Sample from the posterior and see if they fit differently. Take 5-7 samples. Check Cams
* Enny blev glad da vi fik et bedre resultat end onsdagsgruppen
* Antallet af parameter i vores last layer 
  * Hvis vores hessian ikke er stor nok, kan det være at approx af diag er langsommere end at udregne det hele
  * Vi har 100*7=700 parametre, og det er ikke helt så mange. 
  * Could might be an explanation. 
  * Kron and full might be the same results, since the kron is only given the latest layer. 
* Anden 
  * Deep ensample
  * Laver en fordeling over de forskellige vægte fra måske 5 netværks.
  * Midler over predictions.
  * Efter softmax, tager man avg af deres propbabilities. 
  * Typisk store accuracy increases
  * Som kontrast til Laplace
  * Træn måske 10 modeller. Tjek Ensample med 3, 5 og 10 perhabs. For at se en forbedring. 
  * Randomized initialisation - Random seed -  change that
* gradiant explanation eller deep ensample. 
  * Forskellige måder at sammenligne på.
  * Overvejelser hvad man kunne have gjort i stedet for. 
  * Gradiant gemmer vi til efter påsken. 
* 7 samlpes fra posterior. Nogen af dem forkusere måske ikke på bogstaverne. Lav 'intersektion' på tværs og se hvad der bliver fokuseret på
  * Take min values of the explanation 'vector'/stack
  * Union max of the stack. 
* Rapporten bliver evalueret. Brug mest tid på den. Resultater er kun gode til at skabe en bedre rapport. 
* Powerpoint af resultater og rapport. Alle forklarer lidt om det vi har lavet. 
* Spørgsmål kan godt være på baggrund af rapporten. 
* Ting i fremlæggelsen skal gerne være i rapporten.
* Tjek op med Søren/Michael evt. med appendix og billeder i rapporten ifh til 4 sider. 