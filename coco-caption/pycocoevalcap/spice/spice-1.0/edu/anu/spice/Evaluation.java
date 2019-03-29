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

import org.json.simple.JSONAware;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;

import edu.anu.spice.TupleSet.Count;

public class Evaluation implements JSONAware {
	public double f; 	// f-1 score
	public double pr; 	// precision
	public double re; 	// recall
	public int tp; 		// true positives
	public int fp; 		// false positives
	public int fn; 		// false negatives
	public int numImages;

	public Evaluation() {
		f = 0.0;
		pr = 0.0;
		re = 0.0;
		tp = 0;
		fp = 0;
		fn = 0;	
		numImages = 0;
	}
	
	protected void calcFScore(boolean allowNan){
		double beta = 1.0;
		int ref_n = this.fn + this.tp;
		int test_n = this.fp + this.tp;
		if (ref_n > 0) {
			this.re = (double)tp / (double)ref_n;
			if (test_n > 0) {
				this.pr = (double)tp / (double)test_n;
			}
			if (this.pr > 0 && this.re > 0) {
				this.f = (1.0 + beta * beta) * (this.pr * this.re) / (beta * beta * this.pr + this.re);
			}
		}
		if (allowNan && ref_n == 0) {
			this.f = Double.NaN;
			this.pr = Double.NaN;
			this.re = Double.NaN;
		}
	}

	public Evaluation(TupleSet candidates, TupleSet references, boolean allowNan, boolean useSynsets) {
		this();
		this.numImages = 1;
		Count intersection;
		if (useSynsets){
			intersection = candidates.match_similar(references);
		} else {
			intersection = candidates.match_exact(references);
		}
		this.tp = intersection.n;
		this.fp = candidates.size() - this.tp;
		this.fn = references.size() - this.tp;
		this.calcFScore(allowNan);
	}

	public String toString(String delim) {
		StringBuilder sb = new StringBuilder();
		sb.append(f + delim);
		sb.append(pr + delim);
		sb.append(re + delim);
		return sb.toString().trim();
	}

	public String toString() {
		return this.toString(",");
	}

	@SuppressWarnings("unchecked")
	@Override
	public String toJSONString() {
		JSONObject jsonObj = new JSONObject();
		jsonObj.put("tp", new Integer(tp));
		jsonObj.put("fp", new Integer(fp));
		jsonObj.put("fn", new Integer(fn));		
		jsonObj.put("f", new Double(f));
		jsonObj.put("pr", new Double(pr));
		jsonObj.put("re", new Double(re));
		jsonObj.put("numImages", new Integer(numImages));
		return JSONValue.toJSONString(jsonObj);
	}

}
