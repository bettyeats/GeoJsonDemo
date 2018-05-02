namespace java classifyservice

struct RST_RCGZ{
1: bool bret,
2: list<string> type,
3: list<double> val,
4: list<double> features,
5: string message,
}

service Service{
RST_RCGZ recognize_Image(1: binary img),
}