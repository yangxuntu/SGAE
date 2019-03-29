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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.json.simple.JSONArray;
import org.json.simple.JSONAware;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;

/**
 * Class used to hold SPICE statistics, including final score
 * 
 */
public class SpiceStats implements JSONAware {

	protected List<Object> imageIds;
	protected Boolean isDetailed;
	protected List<Map<String, Evaluation>> scores;
	protected List<TupleSet> testTuples;
	protected List<TupleSet> refTuples;
	protected Map<String, TupleFilter> filters;
	protected double SPICE;

	public SpiceStats(Map<String, TupleFilter> filters, Boolean isDetailed) {
		this.isDetailed = isDetailed;
		this.imageIds = new ArrayList<Object>();
		this.scores = new ArrayList<Map<String, Evaluation>>();
		this.testTuples = new ArrayList<TupleSet>();
		this.refTuples = new ArrayList<TupleSet>();
		this.filters = filters;
	}
	
	public void score(Object object, SceneGraph test, SceneGraph ref, boolean useSynsets){
		this.imageIds.add(object);
		TupleSet testT = new TupleSet(test);
		TupleSet refT = new TupleSet(ref);
		Evaluation all = new Evaluation(testT, refT, false, useSynsets);
		if (this.isDetailed){
			this.testTuples.add(testT);
			this.refTuples.add(refT);			
		}
		HashMap<String, Evaluation> score = new HashMap<String, Evaluation>();
		score.put("All", all);
		for (Entry<String, TupleFilter> pair: filters.entrySet()){
	    	TupleSet testF = new TupleSet(test, pair.getValue());
	    	TupleSet refF = new TupleSet(ref, pair.getValue());
	        score.put(pair.getKey(), new Evaluation(testF, refF, true, useSynsets));
	    }
	    this.scores.add(score);
	}
	
	private Evaluation macroAverage(String filter){
		Evaluation result = new Evaluation();
		int imageCount = 0;
		for (Map<String, Evaluation> score: this.scores){
			Evaluation s = score.get(filter);
			result.tp += s.tp;
			result.fp += s.fp;
			result.fn += s.fn;
			if (!Double.isNaN(s.f) && !Double.isNaN(s.pr) && !Double.isNaN(s.re)){
				result.f += s.f;
				result.pr += s.pr;
				result.re += s.re;
				imageCount += 1;
			}
		}
		if (imageCount > 0){
			result.f /= (double) imageCount;
			result.pr /= (double) imageCount;
			result.re /= (double) imageCount;
			result.numImages = imageCount;
		} else {
			result.f = Double.NaN;
			result.pr = Double.NaN;
			result.re = Double.NaN;
			result.numImages = 0;
		}
		return result;
	}
	
	private Evaluation microAverage(String filter){
		Evaluation result = this.macroAverage(filter);
		result.calcFScore(false);
		return result;
	}	
	
	protected String toString(Evaluation spice){
		String output = new String();
		output += String.format("  f-score:\t%.3f (SPICE metric)\n", spice.f);
		output += String.format("  precision:\t%.3f\n", spice.pr);
		output += String.format("  recall:\t%.3f\n", spice.re);
		output += String.format("  true pos:\t%d\n", spice.tp);
		output += String.format("  false pos:\t%d\n", spice.fp);
		output += String.format("  false neg:\t%d\n", spice.fn);
		output += String.format("  num images:\t%d\n", spice.numImages);
		return output;
	}

	@Override		
	public String toString() {
		String output = new String();
		output += "********  SPICE Evaluation  ********\n";
		output += "\nAll tuples\n";
		output += this.toString(this.macroAverage("All"));
		for (Entry<String, TupleFilter> pair: filters.entrySet()){
			String filter = pair.getKey();
			output += String.format("\n%s tuples\n", filter);
			output += this.toString(this.macroAverage(filter));	
	    }
		return output;
	}
	
	@SuppressWarnings("unchecked")
	public JSONArray toJSONArray(){
		JSONArray array = new JSONArray();
		for (int i=0; i<this.imageIds.size(); ++i){
			JSONObject jsonObj = new JSONObject();
			jsonObj.put("image_id", this.imageIds.get(i));
			jsonObj.put("scores", new JSONObject(this.scores.get(i)));
			if (this.isDetailed) {
				jsonObj.put("test_tuples", this.testTuples.get(i));
				jsonObj.put("ref_tuples", this.refTuples.get(i));
			}
			array.add(jsonObj);
		}
		return array;
	}

	@Override
	public String toJSONString() {
		return JSONValue.toJSONString(this.toJSONArray());
	}

}
