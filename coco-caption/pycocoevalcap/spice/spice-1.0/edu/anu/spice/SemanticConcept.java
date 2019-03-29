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

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import org.json.simple.JSONAware;

import com.google.common.collect.ComparisonChain;
import com.google.common.collect.Ordering;

import edu.stanford.nlp.ling.IndexedWord;

/**
 * SemanticConcepts are a small set of words or phrases and 
 * their associated Wordnet synsets.
 * 
 */
public class SemanticConcept implements JSONAware, Comparable<SemanticConcept>, java.io.Serializable {

	protected HashSet<String> concepts; // e.g. word or phrase
	protected HashSet<Integer> synsets;
	private static final long serialVersionUID = 1L;

	public SemanticConcept(String concept, Set<Integer> synsets) {
		this.concepts = new HashSet<String>();
		this.concepts.add(concept);
		this.synsets = new HashSet<Integer>();
		this.synsets.addAll(synsets);
	}

	public SemanticConcept(IndexedWord word, HashSet<Integer> synsets) {
		this(word.lemma().trim().toLowerCase(), synsets);
	}

	public boolean merge(SemanticConcept o) {
		boolean changedConcepts = this.concepts.addAll(o.concepts);
		boolean changedSynsets = this.synsets.addAll(o.synsets);
		return changedConcepts || changedSynsets; 
	}

	@Override
	public int hashCode() {
		return new int[] { this.concepts.hashCode(), this.synsets.hashCode() }.hashCode();
	}

	@Override
	public boolean equals(Object o) {
		if (o == null) {
			return false;
		}
		if (!(o instanceof SemanticConcept)) {
			return false;
		}
		SemanticConcept oReln = (SemanticConcept) o;
		return this.concepts.equals(oReln.concepts) && this.synsets.equals(oReln.synsets);
	}

	@Override
	public int compareTo(SemanticConcept o){
		return ComparisonChain.start()
		    .compare(this.concepts, o.concepts, Ordering.<String>natural().lexicographical())
		    .compare(this.synsets, o.synsets, Ordering.<Integer>natural().lexicographical())
		    .result();
	}

	/**
	 * SemanticConcepts are similar if they share a synset or a concept
	 */
	public boolean similarTo(Object o) {
		if (o == null) {
			return false;
		}
		if (!(o instanceof SemanticConcept)) {
			return false;
		}
		SemanticConcept otherConcept = (SemanticConcept) o;
		HashSet<Integer> synset_intersection = new HashSet<Integer>(this.synsets);
		synset_intersection.retainAll(otherConcept.synsets);
		if (!synset_intersection.isEmpty()){
			return true;
		}
		HashSet<String> concept_intersection = new HashSet<String>(this.concepts);
		concept_intersection.retainAll(otherConcept.concepts);
		return !concept_intersection.isEmpty();
	}

	public float similarity(Object o) {
		if (o == null) {
			return 0;
		}
		if (!(o instanceof SemanticConcept)) {
			return 0;
		}
		SemanticConcept otherConcept = (SemanticConcept) o;
		HashSet<String> concept_intersection = new HashSet<String>(this.concepts);
		concept_intersection.retainAll(otherConcept.concepts);
		if (!concept_intersection.isEmpty()) {
			return 1;
		}
		HashSet<Integer> synset_intersection = new HashSet<Integer>(this.synsets);
		synset_intersection.retainAll(otherConcept.synsets);
		HashSet<Integer> synset_union = new HashSet<Integer>(this.synsets);
		synset_union.addAll(otherConcept.synsets);
		return ((float)synset_intersection.size()) / ((float) synset_union.size());
	}

	@Override
	public String toString() {
		StringBuilder value_list = new StringBuilder();
		for (Iterator<String> iter = this.concepts.iterator(); iter.hasNext();) {
			value_list.append(iter.next());
			if (iter.hasNext()) {
				value_list.append("/");
			}
		}
		return value_list.toString();
	}

	@Override
	public String toJSONString() {
		return this.toString();
	}

}
