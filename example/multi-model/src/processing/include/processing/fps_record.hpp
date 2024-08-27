#pragma once
#include <chrono>
#include <vector>
#include <mutex>

class FpsRecorder{
    public:
    FpsRecorder(size_t interval):interval_{interval},ring_point_{0},acc_{0}{
        last_ = std::chrono::steady_clock::now();
    }
    void record(){
        auto now = std::chrono::steady_clock::now();
        long duration =  (long)std::chrono::duration_cast<std::chrono::milliseconds>(now - last_).count();
        last_ = now;
        if(milliseconds_records_.size() < interval_){
            milliseconds_records_.push_back(duration);
            ring_point_ = (ring_point_+1)%interval_;
            acc_+=duration;
            return;
        }
        acc_ -=milliseconds_records_[ring_point_];
        milliseconds_records_[ring_point_]= duration;
        acc_ += duration;
        ring_point_ = (ring_point_+1)%interval_;
    }

    float fps(){
        return (float)milliseconds_records_.size()/(float)acc_*1000.0f;
    }

    private:
    using ClockType = decltype(std::chrono::steady_clock::now());
    ClockType last_;
    size_t interval_;
    
    std::vector<long> milliseconds_records_;
    size_t ring_point_;
    long acc_;
};
class FpsRecorder_mt{
    public:
    FpsRecorder_mt(size_t interval):interval_{interval},ring_point_{0},acc_{0}{
        last_ = std::chrono::steady_clock::now();
    }
    void record(){
        std::lock_guard<std::mutex> lock(m_);
        auto now = std::chrono::steady_clock::now();
        long duration =  (long)std::chrono::duration_cast<std::chrono::milliseconds>(now - last_).count();
        last_ = now;
        if(milliseconds_records_.size() < interval_){
            milliseconds_records_.push_back(duration);
            ring_point_ = (ring_point_+1)%interval_;
            acc_+=duration;
            return;
        }
        acc_ -=milliseconds_records_[ring_point_];
        milliseconds_records_[ring_point_]= duration;
        acc_ += duration;
        ring_point_ = (ring_point_+1)%interval_;
    }

    float fps(){
        std::lock_guard<std::mutex> lock(m_);
        return (float)milliseconds_records_.size()/(float)acc_*1000.0f;
    }

    private:
    using ClockType = decltype(std::chrono::steady_clock::now());
    ClockType last_;
    size_t interval_;
    
    std::vector<long> milliseconds_records_;
    size_t ring_point_;
    long acc_;
    std::mutex m_;
};