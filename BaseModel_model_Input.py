from pydantic import BaseModel

class model_input(BaseModel):

    tcp_srcport: int
    tcp_dstport: int
    tcp_flags: str
    tcp_ack: int
    tcp_window_size_value: int
    tcp_connection_fin: int
    tcp_connection_syn: int
    tcp_connection_rst: int
    tcp_payload: str  # May have missing or empty values
    ip_src: str       # May have missing or empty values
    ip_dst: str       # May have missing or empty values
    mqtt_clientid: str  # May have missing or empty values
    mqtt_msgtype: int
    mqtt_topic: str    # May have missing or empty values
    mqtt_kalive: int
    mqtt_len: int
    tcp_checksum: str  # May have missing or empty values
    tcp_hdr_len: int
    frame_time_delta: float
    frame_time_relative: float
    tcp_time_delta: float