#include <omp.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <set>
#include <string>
#include <cmath>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <pybind11/stl.h>
#include <random>
namespace py = pybind11;
using namespace mlpack::kmeans;

static const float SIGMA = 0.09; //(2*0.14**2)


std::vector<std::vector<int>> find_neigh(py::array_t<double> accumulated_pointcloud, double voxel_size, int size_curr, std::string name){
    std::vector<std::vector<int>> neighbors;
    py::buffer_info buf1 = accumulated_pointcloud.request();
    std::vector<int> static_value;
    if (name=="nuscenes"){
        static_value = {0,7,10,11,12,13,14,15};
    }
    if (name=="semantickitti"){
        //static_value = {0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
        static_value = {8,9,10,11,12,13,14,15,16,17,18};
    }
    double *ptr1 = (double *) buf1.ptr;	
    int Y = buf1.shape[1];
    int X = buf1.shape[0];
    //int offsets[2] = {-1,1};
    //Get limit of pointcloud
    double minX = ptr1[0]; double minY = ptr1[1]; double minZ = ptr1[2]; double maxX = ptr1[0]; double maxY = ptr1[1]; double maxZ = ptr1[2];
    for(int i=1; i<X; i++){
        if(ptr1[i*Y]>maxX) maxX = ptr1[i*Y];
        if(ptr1[i*Y+1]>maxY) maxY = ptr1[i*Y+1];
        if(ptr1[i*Y+2]>maxZ) maxZ = ptr1[i*Y+2];
        if(ptr1[i*Y]<minX) minX = ptr1[i*Y];
        if(ptr1[i*Y+1]<minY) minY = ptr1[i*Y+1];
        if(ptr1[i*Y+2]<minZ) minZ = ptr1[i*Y+2];
    }

	long sampleNX = (int)floor((maxX - floor(minX * (1/voxel_size)) * voxel_size) / voxel_size) + 2;
	long sampleNY = (int)floor((maxY - floor(minY * (1/voxel_size)) * voxel_size) / voxel_size) + 2;

    //Construct the voxel map
    std::unordered_map<long int,std::vector<int>> voxel_map;
    std::unordered_map<long int,std::vector<int>> voxel_map_curr;
    long iX, iY, iZ;
    long int mapIdx;
    for(int i=0; i<X-size_curr; i++){
        //auto is_in = std::find(std::begin(static_value), std::end(static_value), ptr2[i]);
        // When the element is not found, std::find returns the end of the range
        //if (is_in != std::end(static_value)) {
        iX = (long)floor((ptr1[i*Y]-minX) / voxel_size);
        iY = (long)floor((ptr1[i*Y+1]-minY) / voxel_size);
        iZ = (long)floor((ptr1[i*Y+2]-minZ) / voxel_size);
        mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;
        if (voxel_map.count(mapIdx) < 1)
            voxel_map.emplace(mapIdx, std::vector<int>{i});
        else{
            voxel_map[mapIdx].push_back(i);
        //    }
        }
    }

    std::vector<long> indices;

    for(int i=X-size_curr; i<X; i++){
        iX = (long)floor((ptr1[i*Y]-minX) / voxel_size);
        iY = (long)floor((ptr1[i*Y+1]-minY) / voxel_size);
        iZ = (long)floor((ptr1[i*Y+2]-minZ) / voxel_size);
		mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;
		if (voxel_map_curr.count(mapIdx) < 1){
			voxel_map_curr.emplace(mapIdx, std::vector<int>{i});
            indices.push_back(mapIdx);
        }
        else{
            voxel_map_curr[mapIdx].push_back(i);
        }
    }

    for(size_t i=0;i<indices.size();i++) {
        
        mapIdx = indices.at(i);
        std::vector<int> points_inside = voxel_map_curr[mapIdx];
        std::vector<int> neighbor_points;
        iZ = (long)mapIdx/(sampleNX*sampleNY);
        iY = (long)(mapIdx-iZ*sampleNX*sampleNY)/sampleNX;
        iX = (long)mapIdx%sampleNX;
        if (voxel_map.count(mapIdx) > 0){
            neighbor_points.insert(std::end(neighbor_points), std::begin(voxel_map[mapIdx]), std::end(voxel_map[mapIdx]));
        }
        for (int offsetx = -1; offsetx<2; offsetx++){
            if (iX + offsetx < 0 || iX + offsetx > sampleNX) continue;
            for (int offsety = -1; offsety<2; offsety++){
                if (iY + offsety < 0 || iY + offsety > sampleNY) continue;
                if (offsetx != 0 && offsety !=0) continue;
                for (int offsetz = -1; offsetz<2; offsetz++){
                    if (offsetx != 0 && offsetz !=0) continue;
                    if (offsety != 0 && offsetz !=0) continue;
                    if (iZ + offsetz < 0) continue;
                    long int mapIdxNeigbhor;
                    mapIdxNeigbhor = iX + offsetx + sampleNX*(iY+offsety) + sampleNX*sampleNY*(iZ+offsetz);
                    if (voxel_map.count(mapIdxNeigbhor) > 0){
                        neighbor_points.insert(std::end(neighbor_points), std::begin(voxel_map[mapIdxNeigbhor]), std::end(voxel_map[mapIdxNeigbhor]));
                    }
                }
            }
        }

        if(neighbor_points.size() > 0){
            neighbors.push_back(neighbor_points);
            neighbors.push_back(points_inside);
        }

    }

    return neighbors;
}


std::pair<py::array_t<int>,py::array_t<double>> compute_labels(py::array_t<double> accumulated_pointcloud, py::array_t<int> accumulated_labels, py::array_t<double> accumulated_confidence, int size_curr, double voxel_size, int n_labels, std::string name, float dist_charac){
    py::buffer_info buf1 = accumulated_pointcloud.request();
    py::buffer_info buf2 = accumulated_labels.request();
    py::buffer_info buf3 = accumulated_confidence.request();
    double *ptr1 = (double *) buf1.ptr;	
    double *ptr3 = (double *) buf3.ptr;	
    double sigma = pow(dist_charac,2)/log(2);
    int *ptr2 = (int *) buf2.ptr;
    int Y = buf1.shape[1];
    std::vector<std::vector<int>> neighbors = find_neigh(accumulated_pointcloud, voxel_size, size_curr, name);
    if (neighbors.size()==0){
        return std::make_pair(accumulated_labels,accumulated_confidence);
    }
    #pragma omp parallel for num_threads(16)
    for (unsigned int idx = 0; idx < neighbors.size()/2; idx++){
        std::vector<int> bufn1 = neighbors[2*idx];
        std::vector<int> bufn2 = neighbors[2*idx+1];
        std::vector<std::vector<double>> neighbors_points;
        std::vector<int> neighbors_labels;
        for (unsigned int idxn = 0; idxn < bufn1.size(); idxn++){
            int current_index = bufn1[idxn];
            std::vector<double> local_point{ptr1[current_index*Y],ptr1[current_index*Y+1],ptr1[current_index*Y+2],ptr3[current_index]};
            neighbors_points.push_back(local_point);
            neighbors_labels.push_back(ptr2[current_index]);

        }

        for (unsigned int idxn = 0; idxn < bufn2.size(); idxn++){
            int current_index = bufn2[idxn];
            if (current_index<Y-size_curr) {
                continue;
            }
            double local_ponderation[n_labels+1] = {};
            double sum_conf[n_labels+1] = {};
            double n_contrib[n_labels+1] = {};
            for(std::size_t i = 0; i < neighbors_points.size(); ++i) {
                double dist = sqrt(pow(neighbors_points.at(i).at(0)-ptr1[current_index*Y],2)+pow(neighbors_points.at(i).at(1)-ptr1[current_index*Y+1],2)+pow(neighbors_points.at(i).at(2)-ptr1[current_index*Y+2],2));
                float weight = exp(-pow(dist,2)/sigma)*neighbors_points.at(i).at(3);
                if (weight > 0.5){
                    local_ponderation[neighbors_labels.at(i)+1] += weight;
                    sum_conf[neighbors_labels.at(i)+1] += neighbors_points.at(i).at(3);
                    n_contrib[neighbors_labels.at(i)+1] +=1;
                }
            }
            double max = 0;
            for (int i =0; i<n_labels+1; i++){
                if (local_ponderation[i] > max){
                    ptr2[current_index] = i-1;
                    max = local_ponderation[i];
                    ptr3[current_index] = sum_conf[i]/n_contrib[i];
                }
            }
        }
    }

    return std::make_pair(accumulated_labels,accumulated_confidence);
        
}

std::vector<std::vector<int>> make_clusters(py::array_t<double> accumulated_pointcloud, py::array_t<int> accumulated_labels, int size_curr, double voxel_size, int n_cluster, std::string cluster_method){
    py::buffer_info buf1 = accumulated_pointcloud.request();
    py::buffer_info buf2 = accumulated_labels.request();
    double *ptr1 = (double *) buf1.ptr;	
    int *ptr2 = (int *) buf2.ptr;
    int Y = buf1.shape[1];
    int X = buf1.shape[0];
    std::vector<int> remaining_points;
    std::vector<std::unordered_set<int>> clusters_vox;
    std::vector<std::vector<int>> clusters;
    for(int i =0; i<n_cluster;i++){
        clusters.push_back(std::vector<int>());
        clusters_vox.push_back(std::unordered_set<int>());
    }
    for(int i=X-size_curr; i<X; i++){
        if (ptr2[i] == -1){
            remaining_points.push_back(i);
        } 


    }
    if (cluster_method == "Kmeans"){
    
        arma::mat data(2,remaining_points.size());
        for(size_t i=0; i<remaining_points.size(); i++){
            int idx = remaining_points[i];
            data(0,i) = ptr1[Y*idx];
            data(1,i) = ptr1[Y*idx+1];
            //data(2,i) = ptr1[Y*idx+2];
        }

        arma::Row<size_t> assignments;
        KMeans<> k(20);
        k.Cluster(data, n_cluster, assignments);

        //Get limit of pointcloud
        double minX = ptr1[0]; double minY = ptr1[1]; double minZ = ptr1[2]; double maxX = ptr1[0]; double maxY = ptr1[1]; double maxZ = ptr1[2];
        for(int i=1; i<X; i++){
            if(ptr1[i*Y]>maxX) maxX = ptr1[i*Y];
            if(ptr1[i*Y+1]>maxY) maxY = ptr1[i*Y+1];
            if(ptr1[i*Y+2]>maxZ) maxZ = ptr1[i*Y+2];
            if(ptr1[i*Y]<minX) minX = ptr1[i*Y];
            if(ptr1[i*Y+1]<minY) minY = ptr1[i*Y+1];
            if(ptr1[i*Y+2]<minZ) minZ = ptr1[i*Y+2];
        }

        int sampleNX = (int)floor((maxX - floor(minX * (1/voxel_size)) * voxel_size) / voxel_size) + 2;
        int sampleNY = (int)floor((maxY - floor(minY * (1/voxel_size)) * voxel_size) / voxel_size) + 2;
        //Construct the voxel map
        std::unordered_map<long int,std::vector<int>> voxel_map;
        int iX, iY, iZ;
        long int mapIdx;
        for(int i=0; i<X; i++){
            iX = (int)floor((ptr1[i*Y]-minX) / voxel_size);
            iY = (int)floor((ptr1[i*Y+1]-minY) / voxel_size);
            iZ = (int)floor((ptr1[i*Y+2]-minZ) / (2*voxel_size));
            mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;
            if (voxel_map.count(mapIdx) < 1)
                voxel_map.emplace(mapIdx, std::vector<int>{i});
            else{
                voxel_map[mapIdx].push_back(i);
            }
        }
        for (size_t i=0; i<assignments.size();i++){
            size_t cluster_number = assignments[i];
            int idx = remaining_points[i];
            iX = (int)floor((ptr1[idx*Y]-minX) / voxel_size);
            iY = (int)floor((ptr1[idx*Y+1]-minY) / voxel_size);
            iZ = (int)floor((ptr1[idx*Y+2]-minZ) / (2*voxel_size));
            mapIdx = iX + sampleNX*iY+ sampleNX*sampleNY*iZ;
            std::vector<int> list_indices;
            list_indices.push_back(mapIdx);
            if ((ptr1[idx*Y]-minX) / voxel_size - (float)iX > 0.66 && iX + 1 < sampleNX) {
                list_indices.push_back(iX+1 + sampleNX*iY+ sampleNX*sampleNY*iZ);
                if ((ptr1[idx*Y+1]-minY) / voxel_size - (float)iY > 0.66 && iY + 1 < sampleNY)  {
                    list_indices.push_back(iX+1 + sampleNX*(iY+1)+ sampleNX*sampleNY*iZ);
                }
                if ((ptr1[idx*Y+1]-minY) / voxel_size - (float)iY < 0.33 && iY - 1 > 0){
                    list_indices.push_back(iX+1 + sampleNX*(iY-1)+ sampleNX*sampleNY*iZ);
                }
            }
            if ((ptr1[idx*Y]-minX) / voxel_size - (float)iX < 0.33 && iX-1 > 0) {
                list_indices.push_back(iX-1 + sampleNX*iY+ sampleNX*sampleNY*iZ);
                if ((ptr1[idx*Y+1]-minY) / voxel_size - (float)iY > 0.66 && iY + 1 < sampleNY) {
                    list_indices.push_back(iX-1 + sampleNX*(iY+1)+ sampleNX*sampleNY*iZ);
                }
                
                if ((ptr1[idx*Y+1]-minY) / voxel_size - (float)iY < 0.33 && iY - 1 > 0 ) {
                    list_indices.push_back(iX-1 + sampleNX*(iY-1)+ sampleNX*sampleNY*iZ);
                }
            }
            if ((ptr1[idx*Y+1]-minY) / voxel_size - (float)iY > 0.66 && iY + 1 < sampleNY) {
                list_indices.push_back(iX + sampleNX*(iY+1)+ sampleNX*sampleNY*iZ);
            }
            if ((ptr1[idx*Y+1]-minY) / voxel_size - (float)iY < 0.33&& iY - 1> 0) {
                list_indices.push_back(iX + sampleNX*(iY-1)+ sampleNX*sampleNY*iZ);
            }
            for(std::size_t k = 0; k < list_indices.size(); ++k) {
                mapIdx = list_indices.at(k);
                if (voxel_map.count(mapIdx) >= 1){
                    clusters_vox[cluster_number].insert(mapIdx);
                }
                if ((ptr1[idx*Y+2]-minZ) / (2*voxel_size) - (float)iZ < 0.33 && iZ  > 0){
                    if (voxel_map.count(mapIdx-sampleNX*sampleNY) >= 1){
                        clusters_vox[cluster_number].insert(mapIdx-sampleNX*sampleNY);
                    }
                }
                if ((ptr1[idx*Y+2]-minZ) / (2*voxel_size) - (float)iZ > 0.66){
                    if (voxel_map.count(mapIdx+sampleNX*sampleNY) >= 1){
                        clusters_vox[cluster_number].insert(mapIdx+sampleNX*sampleNY);
                    }
                }
            }

        }


        for (size_t i=0; i<clusters_vox.size();i++){
            for (const auto& elem: clusters_vox[i]) {
                {
                    clusters[i].insert(std::end(clusters[i]), std::begin(voxel_map[elem]), std::end(voxel_map[elem]));
                }
            }
        }
    }
    
    return clusters;
        
}


PYBIND11_MODULE(propagation, m) {
    m.doc() = "pybind11 rewriting of propagation functions";
    m.def("compute_labels", &compute_labels);
    m.def("cluster", &make_clusters);
}