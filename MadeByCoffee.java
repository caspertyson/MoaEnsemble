/*
 *    OzaBag.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.classifiers.meta;

import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import com.github.javacliparser.IntOption;

/**
 * Incremental on-line bagging of Oza and Russell.
 *
 * <p>Oza and Russell developed online versions of bagging and boosting for
 * Data Streams. They show how the process of sampling bootstrap replicates
 * from training data can be simulated in a data stream context. They observe
 * that the probability that any individual example will be chosen for a
 * replicate tends to a Poisson(1) distribution.</p>
 *
 * <p>[OR] N. Oza and S. Russell. Online bagging and boosting.
 * In Artiﬁcial Intelligence and Statistics 2001, pages 105–112.
 * Morgan Kaufmann, 2001.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classiﬁer to train</li>
 * <li>-s : The number of models in the bag</li> </ul>
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class MadeByCoffee extends AbstractClassifier implements MultiClassClassifier,
        CapabilitiesHandler {

    @Override
    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    protected Classifier[] ensemble;

    protected Classifier candidate;

    protected double[][] performance; //[0] = numCorrect, [1] = totalPredictions

    protected double[] candidatePerformance; //[0] = numCorrect, [1] = totalPredictions

    protected int windowSize = 1000;

    protected int instanceNum = 0;

    @Override
    public void resetLearningImpl() {
        this.performance = new double[this.ensembleSizeOption.getValue()][2];
        this.candidatePerformance = new double[2];
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
        this.candidate = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        this.candidate.resetLearning();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        for (int i = 0; i < this.ensemble.length; i++) {
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
        }
        // Train Candidate
        int k = MiscUtils.poisson(1.0, this.classifierRandom);
        if (k > 0) {
            Instance weightedInst = (Instance) inst.copy();
            weightedInst.setWeight(inst.weight() * k);
            this.candidate.trainOnInstance(weightedInst);
        }
    }
    private void replaceWorstAndReset(){
        // find worst performer
        int worstPerformerIndex = 0;
        double currentPerformance = 0;
        for(int p = 0; p < this.performance.length; p++){
            currentPerformance = this.performance[p][0] / this.performance[p][1];
            if(this.performance[worstPerformerIndex][0]/ this.performance[worstPerformerIndex][1]
                    > currentPerformance){
                worstPerformerIndex = p;
            }
        }

        // replace worst with candidate if necessary
        if(this.candidatePerformance[0] / this.candidatePerformance[1]
                > this.performance[worstPerformerIndex][0] / this.performance[worstPerformerIndex][1]){
            this.ensemble[worstPerformerIndex] = this.candidate;
            this.performance[worstPerformerIndex][0] = this.candidatePerformance[0];
            this.performance[worstPerformerIndex][1] = this.candidatePerformance[1];
        }
        this.candidate = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        this.candidate.resetLearning();
        this.candidatePerformance = new double[2];
        // reset candidate, candidatePerformance and performance at the correct index
        this.instanceNum = 0;
    }
    @Override
    public double[] getVotesForInstance(Instance inst) {
        this.instanceNum++;
        // If end of window
        if(this.instanceNum % this.windowSize == 0){
            replaceWorstAndReset();
        }
        // get votes through performance
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));

            if (vote.sumOfValues() > 0.0) {
                for(int y = 0; y < vote.numValues(); y++){
                    combinedVote.setValue(y, 0.0);
                }
                //then cast vote with learner performance: numCorrect/totalPredictions
                combinedVote.setValue(vote.maxIndex(), this.performance[i][0] / this.performance[i][1]);
                combinedVote.addValues(vote);
                if(this.ensemble[i].correctlyClassifies(inst)){
                    this.performance[i][0]++; // correct counter
                }
                this.performance[i][1]++; // total predictions counter
            }
        }
        if(this.candidate.correctlyClassifies(inst)){
            this.candidatePerformance[0]++; // correct counter
        }
        this.candidatePerformance[1]++;// total predictions counter

        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                this.ensemble != null ? this.ensemble.length : 0)};
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.ensemble.clone();
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == MadeByCoffee.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
