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

import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Map.Entry;

import org.fusesource.lmdbjni.Database;
import org.fusesource.lmdbjni.Env;
import org.fusesource.lmdbjni.LMDBException;
import org.fusesource.lmdbjni.Transaction;
import org.nustaq.serialization.FSTConfiguration;

import com.google.common.base.MoreObjects;
import com.google.common.base.Stopwatch;

public class LmdbTupleDB {

	protected static FSTConfiguration conf = FSTConfiguration.createDefaultConfiguration();
	protected String dbPath;

	public LmdbTupleDB(String dbPath) {
		// register most frequently used classes on conf
		conf.registerClass(String.class,ArrayList.class);
		this.dbPath = dbPath;
	}

	@SuppressWarnings("unchecked")
	public ArrayList<ArrayList<String>> get(String caption) {
		try (Env env = new Env()) {
			env.open(this.dbPath);
			try (Database db = env.openDatabase()) {
				byte[] val = db.get(conf.asByteArray(caption));
				if (val == null) {
					return null;
				} else {
					return (ArrayList<ArrayList<String>>)conf.asObject(val);
				}
			}
		}
	}

	public void putTransaction(Map<String, ArrayList<ArrayList<String>>> captionsToTuples) {
		//Stopwatch timer = Stopwatch.createStarted();
		try (Env env = new Env()) {
			env.setMapSize(10000L*2560*4096); // 100GB
			env.open(this.dbPath);
			try (Database db = env.openDatabase()) {
				try (Transaction tx = env.createWriteTransaction()) {					
					for(Entry<String, ArrayList<ArrayList<String>>> item: captionsToTuples.entrySet()) {
						try {
							db.put(tx, conf.asByteArray(item.getKey()), conf.asByteArray(item.getValue()));
						}
						catch (LMDBException ex){
							System.err.println(String.format("Error: Could not cache item to %s with key:\n\"%s\"\nCaption may be too long",
									MoreObjects.firstNonNull(this.dbPath, "NULL"), item.getKey()));
						}
					}
					tx.commit();  // if commit is not called, the transaction is aborted
				}
			}
		}
		//System.out.println("putTransaction took: " + timer.stop());
	}

	@SuppressWarnings("unchecked")
	public Map<String, ArrayList<ArrayList<String>>> getTransaction(List<String> captions) {
		//Stopwatch timer = Stopwatch.createStarted();
		Map<String, ArrayList<ArrayList<String>>> results = new HashMap<String, ArrayList<ArrayList<String>>>();
		try (Env env = new Env()) {
			env.open(this.dbPath);
			try (Database db = env.openDatabase()) {
				for (String caption: captions){
					byte[] val = db.get(conf.asByteArray(caption));
					if (val != null) {
						results.put(caption, (ArrayList<ArrayList<String>>)conf.asObject(val));
					}
				}
			}
		}
		//System.out.println("getTransaction took: " + timer.stop());
		return results;
	}

}
