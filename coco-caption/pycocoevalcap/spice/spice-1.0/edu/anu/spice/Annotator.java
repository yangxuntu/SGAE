/*
 * Copyright (c) 2016, Peter Anderson <peter.anderson@anu.edu.au>
 *
 * This file is part of Semantic Propositional Image Caption Evaluation
 * (SPICE).
 * 
 * SPICE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * SPICE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.

 * You should have received a copy of the GNU Affero General Public
 * License along with SPICE.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

package edu.anu.spice;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

/**
 * Class to perform batched multi-threaded dependency parsing.
 */
public class Annotator {
	
	protected StanfordCoreNLP pipeline;
	protected int numThreads;
	protected int maxBatchSize;
	protected List<List<String>> batches;
	protected int batchNum;

	public Annotator(int numThreads, int batchSize) {
		this.pipeline = null;
		this.numThreads = numThreads;
		this.batches = null;
		this.batchNum = 0;
		this.maxBatchSize = batchSize;
	}
	
	private void initPipeline() {
		System.err.println("Initiating Stanford parsing pipeline");
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,parse,lemma,ner");
		props.setProperty("depparse.extradependencies", "MAXIMAL");
		props.setProperty("threads", Integer.toString(this.numThreads));
		this.pipeline = new StanfordCoreNLP(props);
	}
	
	public void setInput(List<String> captions){
		this.batches = Lists.partition(captions, this.maxBatchSize);
		this.batchNum = 0;
	}
	
	List<Annotation> parseNextBatch(){
		//Stopwatch timer = Stopwatch.createStarted();		
		List<Annotation> result = new ArrayList<Annotation>();
		if (this.batches != null && this.batchNum < this.batches.size()){
			for (String caption : this.batches.get(batchNum)) {
				result.add(new Annotation(caption));
			}
			if (!result.isEmpty()){
				if (this.pipeline == null) {
					initPipeline();
				}
				this.pipeline.annotate(result);
			}
			batchNum++;
			if (batchNum >= this.batches.size()){
				this.batches = null;
			}
		}	
		//System.out.println("parseCaptions took: " + timer.stop());
		return result;
	}
 }
