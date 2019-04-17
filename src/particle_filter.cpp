#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

/**Initialization Step*/
	num_particles = 100; 
	std::default_random_engine gen;
	//create normal distributions to sample from
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	//sample n particles at Gaussian distribution for first position
	for (int i = 0; i < num_particles; ++i) {
		Particle p_temp; 
		p_temp.id = i;
		p_temp.x = dist_x(gen);
		p_temp.y = dist_y(gen);
		p_temp.theta = dist_theta(gen);   
		p_temp.weight = 1.0;
		particles.push_back(p_temp);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

/** Prediction step**/
	std::default_random_engine gen;
	//Create normal distributions for each measurement
	std::normal_distribution<double> dist_x_pos(x, std_pos[0]);
	std::normal_distribution<double> dist_y_pos(y, std_pos[1]);
	std::normal_distribution<double> dist_theta_pos(theta, std_pos[2]);
	//Predict the new x, y, and theta for the next measurement 
	for (int i=0; i<num_particles;i++){
		//If yaw rate is too small, keep theta, and calculate w/o yaw rate
		if(fabs(yaw_rate) < 0.00001) {  
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		//Predict x, y, and theta normally
		else {
			particles[i].x += velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
			particles[i].theta += yaw_rate*delta_t;
		}
		//Add noise to each measurement
		particles[i].y += dist_y_pos(gen);        
		particles[i].x += dist_x_pos(gen);
		particles[i].theta += dist_theta_pos(gen);
	} 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
	
/**Update Step: Association*/
	for (int i=0; i<observations.size();i++){
		//look at the distance to each landmark
		int ident = -1;
		for (int j=0; j<predicted.size(); j++){
			double min = 100000000;
			double distance = helper_functions::dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y);
			//Keep track of the shortest distance and its landmark ID
			if (distance < min){
				min = distance; 
				ident = predicted[j].id;
			}   
		}
		observations[i].id = ident;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  
/**Update Step*/
  
	for (int i=0;i<num_particles;i++)){
		//create the list of landmarks the particle should see within range of the sensor
		vector<LandmarkObs> predicted;
		for (int k=0; k<map_landmarks.landmark_list.size(),k++){
			//calculate the 2D distance between the observation and the landmark
			double distance = helper_functions::dist(map_landmarks.landmark_list[k].x, map_landmarks.landmark_list[k].y, particles[i].x, particles[i].y);
			//if landmark is in range of particle, store it
			if (distance <= sensor_range){
				predicted.push_back(map_landmarks.landmark_list[k]);
			}
		}
      
		//Complete the transformation from observed distances to map coordinate system
		vector<LandmarkObs> trans_ob;	//vector of observations for each particle
		for (int j=0; j<observations.size();j++){
			LandmarkObs observed;
			double delta_theta = particles[i].theta;
			observed.x = particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta) * observations[j].y);
			observed.y = particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)* observations[j].y);
			trans_ob.push_back(observed); //fill the vector using the coordinate transformation-applied observations
		}	
  	
		//Get the landmark ID of the landmark closest to each observation
		dataAssociation(predictions, trans_ob);
      
		//FINALLY: Apply multi-variate gaussian distribution
		//standard deviations of the landmarks
		double std_Lx = std_landmark[0];
		double std_Ly = std_landmark[1];
		particles[i].weight = 1.0;
		//get the position for each observation
		for (j=0; j<trans_ob.size();j++){
			obs_x = trans_ob[j].x;
			obs_y = trans_ob[j].y;
			landmark_id = trans_ob[j].id;
			//get the position for the associated landmark
			for (k=0; k<predictions.size();k++){
				if (predictions[k].id = landmark_id){
					pred_x = predictions[k].x;
					pred_y = prediction[k].y;
				}
			}
			//Multivariate Gaussian probability for that observation given the landmark distance from position
			double obs_w = (1/(2*M_PI*std_Lx*std_Ly)) * exp( -( pow(obs_x-pred_x,2)/(2*pow(std_Lx, 2)) + (pow(obs_y-pred_y,2)/(2*pow(std_Ly, 2)))));

			//Product of this obersvation weight with total observations weight
			particles[i].weight *= obs_w;
		}
	}
}

void ParticleFilter::resample() {

/**Resampling Step*/
	std::discrete_distribution distr;
	//Get weights and max weight.
	vector<double> weights;
	double w_max = 0.0;
	//append the particles weights to a vector "weights" for resampling
	for(int i=0; i<num_particles; i++){
		weights.push_back(particles[i].weight);
		if(particles[i].weight > w_max) {
			w_max = particles[i].weight;
		}
	}

	uniform_real_distribution<double> distDouble(0.0, max_w);
	uniform_int_distribution<int> distInt(0, num_particles - 1);
	int index = distInt(gen);
	double beta = 0.0;
	vector<Particle> resampled[];
	for(int i = 0; i < num_particles; i++) {
		beta += distDouble(gen) * 2.0;
		while(beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled.push_back(particles[index]);
	}
	particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}