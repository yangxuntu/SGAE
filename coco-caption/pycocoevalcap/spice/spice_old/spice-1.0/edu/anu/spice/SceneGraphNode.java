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

/**
*
* Based on the SceneGraphNode in scenegraph-1.0 by Sebastian Schuster.
*
*/

public class SceneGraphNode implements java.io.Serializable {

	private static final long serialVersionUID = 1L;
	protected SemanticConcept object;
	protected Set<SemanticConcept> attributes;

	public SceneGraphNode(SemanticConcept object) {
		this.object = object;
		this.attributes = new HashSet<SemanticConcept>();
		
	}

	public boolean hasAttribute(SemanticConcept word) {
		for (Iterator<SemanticConcept> i = this.attributes.iterator(); i.hasNext();) {
			SemanticConcept attr = i.next();
			if (word.equals(attr)) {
				return true;
			}
		}
		return false;
	}

	public boolean addAttribute(SemanticConcept word) {
		if (this.hasAttribute(word)) {
			return false;
		}
		this.attributes.add(word);
		return true;
	}
	
	public boolean mergeAttribute(SemanticConcept word) {
		if (this.hasAttribute(word)) {
			return false;
		}
		for (Iterator<SemanticConcept> i = this.attributes.iterator(); i.hasNext();) {
			SemanticConcept attr = i.next();
			if (word.similarTo(attr)) {
				attr.merge(word);
				return true;
			}
		}
		this.attributes.add(word);
		return true;
	}

	public Set<SemanticConcept> getAttributes() {
		return this.attributes;
	}

	public SemanticConcept getObject() {
		return this.object;
	}

	@Override
	public boolean equals(Object o) {
		if (o == null) {
			return false;
		}
		if (!(o instanceof SceneGraphNode)) {
			return false;
		}
		SceneGraphNode oNode = (SceneGraphNode) o;
		return this.object.equals(oNode.object) && this.attributes.equals(oNode.attributes);
	}

	public boolean similarTo(Object o) {
		// Dependent on object only, not attributes
		if (o == null) {
			return false;
		}
		if (!(o instanceof SceneGraphNode)) {
			return false;
		}
		SceneGraphNode otherNode = (SceneGraphNode) o;
		return this.object.similarTo(otherNode.object);
	}
	
	public float similarity(Object o) {
		// Dependent on object only, not attributes
		if (o == null) {
			return 0;
		}
		if (!(o instanceof SceneGraphNode)) {
			return 0;
		}
		SceneGraphNode otherNode = (SceneGraphNode) o;
		return this.object.similarity(otherNode.object);
	}

	@Override
	public String toString() {
		return this.object.toString();
	}

	public boolean merge(SceneGraphNode otherNode) {
		boolean changed = this.object.merge(otherNode.object);
		for (SemanticConcept other_attr : otherNode.attributes) {
			changed |= this.mergeAttribute(other_attr);
		}
		return changed;
	}
}
