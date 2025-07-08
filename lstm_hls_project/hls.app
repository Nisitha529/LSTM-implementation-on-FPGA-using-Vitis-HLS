<project xmlns="com.autoesl.autopilot.project" top="top_fpga_lstm" name="lstm_hls_project" projectType="C/C++">
    <includePaths/>
    <libraryPaths/>
    <Simulation>
        <SimFlow name="csim" csimMode="0" lastCsimMode="0"/>
    </Simulation>
    <files xmlns="">
        <file name="lstm_hls_project/src_files/lstm_kernel.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="lstm_hls_project/src_files/lstm_model.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="lstm_hls_project/src_files/lstm_top.hpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="../src_files/tb_lstm.cpp" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
    </files>
    <solutions xmlns="">
        <solution name="solution1" status="active"/>
    </solutions>
</project>

