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
import java.util.Iterator;

import org.json.simple.JSONArray;
import org.json.simple.JSONAware;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;

import com.google.common.collect.ComparisonChain;
import com.google.common.collect.Ordering;

public class SemanticTuple implements JSONAware, Comparable<SemanticTuple> {

	public ArrayList<SemanticConcept> tuple;
	public boolean truthValue;
	public double idf; // Inverse document frequency, or 1 if not in use

	public SemanticTuple(SemanticConcept object) {
		this.tuple = new ArrayList<SemanticConcept>();
		this.tuple.add(object);
		this.truthValue = false;
		this.idf = 1;
	}

	public SemanticTuple(SemanticConcept object, SemanticConcept attribute) {
		this.tuple = new ArrayList<SemanticConcept>();
		this.tuple.add(object);
		this.tuple.add(attribute);
		this.truthValue = false;
		this.idf = 1;
	}

	public SemanticTuple(SemanticConcept object1, SemanticConcept relation, SemanticConcept object2) {
		this.tuple = new ArrayList<SemanticConcept>();
		this.tuple.add(object1);
		this.tuple.add(relation);
		this.tuple.add(object2);
		this.truthValue = false;
		this.idf = 1;
	}

	public int size() {
		return this.tuple.size();
	}

	public SemanticConcept get(int i) {
		return tuple.get(i);
	}

	public boolean similarTo(SemanticTuple o) {
		if (o == null) {
			return false;
		}
		if (this.tuple.size() != o.tuple.size()) {
			return false;
		}
		for (int i = 0; i < this.tuple.size(); i++) {
			if (!this.tuple.get(i).similarTo(o.tuple.get(i))) {
				return false;
			}
		}
		return true;
	}
	
	public ArrayList<ArrayList<String>> enumerateTuples(){
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		if (this.tuple.size() == 1) {
			SemanticConcept object = this.tuple.get(0);
			for (String objectWord : object.concepts){
				ArrayList<String> tuple = new ArrayList<String>();
				tuple.add(objectWord);
				result.add(tuple);
			}
		} else if (this.tuple.size() == 2) {
			SemanticConcept object = this.tuple.get(0);
			SemanticConcept attribute = this.tuple.get(1);
			for (String objectWord : object.concepts){
				for (String attrWord : attribute.concepts){
					ArrayList<String> tuple = new ArrayList<String>();
					tuple.add(objectWord);
					tuple.add(attrWord);
					result.add(tuple);
				}
			}
		} else if (this.tuple.size() == 3) {
			SemanticConcept src = this.tuple.get(0);
			SemanticConcept relation = this.tuple.get(1);
			SemanticConcept tgt = this.tuple.get(2);
			for (String srcWord : src.concepts){
				for (String relWord : relation.concepts){
					for (String tgtWord : tgt.concepts){
						ArrayList<String> tuple = new ArrayList<String>();
						tuple.add(srcWord);
						tuple.add(relWord);
						tuple.add(tgtWord);
						result.add(tuple);
					}
				}
			}
		}
		return result;
	}
	
	public boolean matchesTo(SemanticTuple o) {
		if (o == null) {
			return false;
		}
		if (this.tuple.size() != o.tuple.size()) {
			return false;
		}
		for (int i = 0; i < this.tuple.size(); i++) {
			if (!this.tuple.get(i).equals(o.tuple.get(i))) {
				return false;
			}
		}
		return true;
	}
	
	public void merge(SemanticTuple o) {
		if (this.tuple.size() != o.tuple.size()) {
			return;
		}
		for (int i = 0; i < this.tuple.size(); i++) {
			SemanticConcept merged = this.tuple.get(i);
			merged.merge(o.tuple.get(i));
			this.tuple.set(i, merged);
		}
		this.truthValue = false;
	}
	
	@Override
	public String toString() {
		StringBuilder value_list = new StringBuilder();
		value_list.append("(");
		for (Iterator<SemanticConcept> iter = this.tuple.iterator(); iter.hasNext();) {
			value_list.append(iter.next());
			if (iter.hasNext()) {
				value_list.append(",");
			}
		}
		value_list.append(")");
		return value_list.toString();
	}

	@SuppressWarnings("unchecked")
	@Override
	public String toJSONString() {
		JSONObject jsonObj = new JSONObject();
		JSONArray tuple = new JSONArray();
		for (SemanticConcept concept: this.tuple){
			tuple.add(concept.toJSONString());
		}
		jsonObj.put("tuple", tuple);
		jsonObj.put("truth_value", this.truthValue);
		return JSONValue.toJSONString(jsonObj);
	}

	@Override
	public int compareTo(SemanticTuple o) {
		return ComparisonChain.start()
		    .compareTrueFirst(this.truthValue, o.truthValue)
		    .compare(this.tuple, o.tuple, Ordering.<SemanticConcept>natural().lexicographical())
		    .result();
	}
}
